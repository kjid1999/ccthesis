"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round, get_weights
from diffuseq.text_datasets import load_data_text
from torch.cuda.amp import autocast

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_model_emb,
    load_tokenizer
)
from diffuseq.importance import get_importance_estimate_model

def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0, rejection_rate=0.0, note='none')
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False, start_n=0, )
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@th.no_grad()
def main():
    CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)
    args.device = f"cuda:{CUDA_VISIBLE_DEVICES}"

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
        # dist_util.load_state_dict(args.model_path, False, "model", map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(dist_util.dev())

    tokenizer = load_tokenizer(args)
    # model_emb, tokenizer = load_model_emb(args, tokenizer)
    
    model_emb = th.nn.Embedding(
        num_embeddings=tokenizer.vocab_size, 
        embedding_dim=args.hidden_dim, 
        _weight=model.word_embedding.weight.clone().cpu()
    ).eval().requires_grad_(False)

    model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_emb_copy = get_weights(model_emb, args).eval().requires_grad_(False)

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    ## load data
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(),  # using the same embedding wight with tranining data
        loop=False
    )

    start_t = time.time()

    # batch, cond = next(data_valid)
    # print(batch.shape)

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}_{args.note}.json")
    # fout = open(out_path, 'a')

    all_test_data = []

    idx = 0

    try:
        while True:
            batch, cond = next(data_valid)
            # print(batch.shape)
            if idx % world_size == rank:  # Split data per nodes
                all_test_data.append(cond)
            idx += 1

    except StopIteration:
        print('### End of reading iteration...')
    
    model_emb.to(dist_util.dev())

    if idx % world_size and rank >= idx % world_size:
        all_test_data.append({})  # Dummy data for Remainder : for dist.barrier()

    if rank == 0:
        from tqdm import tqdm
        iterator = tqdm(all_test_data) #, desc='here?' yes
    else:
        iterator = iter(all_test_data)

    for cond in iterator:

        if not cond:  # Barrier for Remainder
            for i in range(world_size):
                dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
            continue

        input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask

        # noise = th.randn_like(x_start) # origin
        noise = model.mean_embed.repeat((*(x_start.shape[:2]), 1)) # I modified
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        x_noised = th.where(input_ids_mask == 0, x_start, noise)

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        see_every_step = True
        if see_every_step:
            print('THIS IS DEBUGGING MODE') 
        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)
        with autocast():
            samples = sample_fn(
                model,
                sample_shape,
                noise=x_noised,
                clip_denoised=args.clip_denoised,
                denoised_fn=partial(denoised_fn_round, args, model_emb),
                model_kwargs=model_kwargs,
                top_p=args.top_p,
                clamp_step=args.clamp_step,
                clamp_first=True,
                mask=input_ids_mask,
                x_start=x_start,
                gap=step_gap,
                collect_every_step=see_every_step # 註解這行來回復原本的code
            )
        # print('samples', len(samples), samples[0].shape)

        # model_emb_copy.cpu()

        # print(samples[0].shape) # samples for each step

        sample = samples # I modified

        if see_every_step:
            #### timesteps flat to batch ####
            # t, bsz, seqlen, vocab -> bsz[i]: t, seqlen, vocab
            sample_id = 1
            mean_embed = model.mean_embed.to(sample[0].device)
            mean_embed = mean_embed[None, :] + 0*sample[0][sample_id]
            sample = th.stack(sample)[:, sample_id]
            sample = th.cat((mean_embed[None], sample))
            ###

        del samples # no help
        # sample = samples[-1] # original, 記憶體洩漏

        # print('decoding for seq2seq', )
        # print(sample.shape)

        if see_every_step:
            logits = model.cpu().get_logits(sample)
        else:
            logits = model.get_logits(sample).to('cpu')  # bsz, seqlen, vocab
        del sample
        cands = th.topk(logits, k=1, dim=-1)

##########
        if see_every_step:
            importance_estimate_model, _ = get_importance_estimate_model()
            my_to_get_evidence(
                sample_id,
                cands,
                input_ids_mask_ori,
                args,
                tokenizer,
                input_ids_x,
                world_size,
                rank,
                importance_estimate_model,
                out_path='generation_outputs/every_step.json'
            )
            # auto exit here

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []

        # tokenizer = load_tokenizer(args)

        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            len_x = args.seq_len - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            # tokens = tokenizer.decode_token(seq)
            len_x = args.seq_len - sum(input_mask).tolist()
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        for i in range(world_size):
            if i == rank:  # Write files sequentially
                fout = open(out_path, 'a')
                for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
                    print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
                fout.close()
            dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')

def my_to_get_evidence(
    sample_id,
    cands,
    input_ids_mask_ori,
    args,
    tokenizer,
    input_ids_x,
    world_size,
    rank,
    importance_estimate_model,
    out_path
    ):
    from diffuseq.importance import importance

    # importance_estimate_model = importance_estimate_model.to('cpu')

    word_lst_recover = []
    word_lst_ref = []
    word_lst_source = []
    importance_lst_recover = []

    input_mask = input_ids_mask_ori[sample_id]
    len_x = args.seq_len - sum(input_mask).tolist()

    seq_input_ids_x = input_ids_x[sample_id]
    source = tokenizer.decode_token(seq_input_ids_x[:len_x])
    ref = tokenizer.decode_token(seq_input_ids_x[len_x:])

    for seq in cands.indices:
        suqeeze_seq = seq[len_x:].squeeze()[None].cuda()
        importance_score = importance(suqeeze_seq, th.full(suqeeze_seq.shape, True, device=suqeeze_seq.device), importance_estimate_model, False)
        importance_lst_recover.append(importance_score)
        # exit()
        tokens = tokenizer.decode_token(seq[len_x:])
        word_lst_recover.append(tokens)

        word_lst_source.append(source)
        word_lst_ref.append(ref)

        if len(word_lst_recover) >= 1500:
            print(tokens)

    for i in range(world_size):
        if i == rank:  # Write files sequentially
            fout = open(out_path, 'w')
            for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
                print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
            fout.close()
        dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

    th.save(importance_lst_recover, 'importance_process.pt')
    print('THIS IS DEBUGGING MODE') 
    exit()

if __name__ == "__main__":
    main()