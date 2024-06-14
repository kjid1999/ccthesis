import sys
sys.path.append('..')

checkpoint_path = r'../diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_maxlen98_lambda0_triplet20240612-21:26:42/ema_0.9999_060000.pt'

import argparse
import json, torch, os
import numpy as np
from basic_utils import (
    create_model_and_diffusion,
    load_defaults_config,
    args_to_dict,
    add_dict_to_argparser
)

def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config('/home/myDiffuSeq/diffuseq/config.json'))
    defaults.update(dict(config_name="bert-base-uncased", vocab_size=30522, learned_mean_embed=True))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    os.environ["LOCAL_RANK"] = '0'

    args = create_argparser().parse_args()
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config('/home/myDiffuSeq/diffuseq/config.json').keys())
    )
    model_para = torch.load(checkpoint_path)
    model.load_state_dict(model_para)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    argsort_idx = diffusion.argsort_idx

    emb_layer = model.word_embedding.weight.detach()[argsort_idx]
    mean_embed = model.mean_embed.detach()

    print(model)
    exit()
    del model
    del diffusion

    print(f'total para. {pytorch_total_params}')
    print(emb_layer)
    print(argsort_idx)
    # print(emb_layer.weight)
    # print(mean_embed)

    word_freq = torch.load(f'../word_freq/bert-base-uncased_qqp.pt')[argsort_idx].numpy()
    argsort_idx = argsort_idx[:(word_freq > 0).sum()]
    emb_layer = emb_layer[word_freq > 0]
    word_freq = np.log(word_freq[word_freq > 0])

    mean_embed = emb_layer.mean(axis=0)

    # save_tsne((emb_layer - mean_embed[None]), word_freq)
    save_umap((emb_layer - mean_embed[None]), word_freq)
    save_mymap((emb_layer - mean_embed[None]), word_freq)

def save_umap(embs, word_freq):
    import umap
    import matplotlib.pyplot as plt
    if not os.path.exists('umap_emb.npy'):
        print('fitting umap_emb.npy')
        umap_model = umap.UMAP(n_components=2)
        embs = embs.numpy()
        X = umap_model.fit_transform(embs)
        with open('umap_emb.npy', 'wb') as f:
            np.save(f, X, allow_pickle=True)
    else:
        print('load umap_emb.npy')
        with open('umap_emb.npy', 'rb') as f:
            X = np.load(f, allow_pickle=True)

    plt.scatter(X[::-1, 0], X[::-1, 1], s=3, c=word_freq[::-1], cmap='Blues')

    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('emb_visualization_umap.png')
    plt.close()

def save_mymap(embs, word_freq):
    import matplotlib.pyplot as plt
    # if not os.path.exists('umap_emb.npy'):
    #     print('fitting umap_emb.npy')
    #     umap_model = umap.UMAP(n_components=2)
    #     embs = embs.numpy()
    #     X = umap_model.fit_transform(embs)
    #     with open('umap_emb.npy', 'wb') as f:
    #         np.save(f, X, allow_pickle=True)
    # else:
    #     print('load umap_emb.npy')
    #     with open('umap_emb.npy', 'rb') as f:
    #         X = np.load(f, allow_pickle=True)


    print(embs.shape)
    embs = embs.numpy()
    radius = np.linalg.norm(embs, axis=-1)
    proj_embs = embs[:, :2]
    X = proj_embs * (radius / np.linalg.norm(proj_embs, axis=-1))[:, None]
    print(X.shape)
    print(type(X))

    plt.scatter(X[::-1, 0], X[::-1, 1], s=3, c=word_freq[::-1], cmap='rainbow')

    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('emb_visualization_mymap.png')
    plt.close()


def save_tsne(embs, word_freq):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    if not os.path.exists('tsne_emb.npy'):
        print('fitting tsne_emb.npy')
        tsne = TSNE(
            init='pca',
            perplexity=40,
            learning_rate=900,
        )
        embs = embs.numpy()
        X = tsne.fit_transform(embs)
        with open('tsne_emb.npy', 'wb') as f:
            np.save(f, X, allow_pickle=True)
    else:
        print('load tsne_emb.npy')
        with open('tsne_emb.npy', 'rb') as f:
            X = np.load(f, allow_pickle=True)

    print(X)
    print(word_freq)

    # X_min, X_max = X.min(0), X.max(0)
    # X = (X - X_min)/(X_max - X_min)
    plt.scatter(X[::-1, 0], X[::-1, 1], s=3, c=word_freq[::-1], cmap='Blues')

    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('emb_visualization_tsne.png')
    plt.close()

    plt.scatter(word_freq, np.linalg.norm(embs, axis=-1), s=3, c='b')
    plt.xlabel('log freq')
    plt.ylabel('dist to anchor')
    plt.savefig('emb_visualization_dis_wrt_freq.png')


if __name__ == "__main__":
    main()
