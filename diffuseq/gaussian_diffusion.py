"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th
import sys
sys.path.append('.')

import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import mean_flat
from .utils.losses import normal_kl, discretized_gaussian_log_likelihood
from .importance import importance, freq_rank, get_importance_estimate_model, calculate_mask_rate, H
from .KL_approximation import _gaussian_extractor, D_var, D_prod

word_freq = th.load(f'/home/myDiffuSeq/word_freq/bert-base-uncased_qqp_nocount_special.pt').numpy()

def triplet_margin_loss(anchor, positive, negative, margin=1):
    '''
    anchor: (D),
    positive: (N1, D),
    negative: (N2, D)
    '''
    dist_pos = (anchor[None] - positive).norm(dim=-1)
    dist_neg = (anchor[None] - negative).norm(dim=-1)

    loss = th.clamp_min(margin + dist_pos[:, None] - dist_neg[None], 0)
    
    return loss.mean()

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, warmup_steps_ratio=0.1):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == 'warmup-decay':
        warmup_steps = max(1, int(warmup_steps_ratio * num_diffusion_timesteps))
        sqrt_steps = get_named_beta_schedule('sqrt', num_diffusion_timesteps)
        beta_mid = sqrt_steps[-warmup_steps]
        warmup = np.linspace(beta_mid, 0.0001, warmup_steps)
        return np.concatenate([sqrt_steps[:-warmup_steps], warmup])
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param predict_xstart: the model outputs to predict x_0, else to predict eps.
    :param learn_sigmas: the model outputs to predict sigma or not. Default: False
    :param rescale_learned_sigmas, sigma_small: details setting of learned sigmas
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        predict_xstart,
        rescale_learned_sigmas,
        learn_sigmas,
        sigma_small,
        use_kl,
        rescale_timesteps=False,
        reg_rate = 0.01,
        rejection_rate=0.,
        denoise=False,
        denoise_rate = 0.2,
        device="",
        max_T = 2000,
        _lambda=0.
    ):
        self.rescale_timesteps = rescale_timesteps
        self.predict_xstart = predict_xstart
        self.rescale_learned_sigmas = rescale_learned_sigmas
        self.learn_sigmas = learn_sigmas
        self.sigma_small = sigma_small
        self.use_kl = use_kl
        self.reg_rate = reg_rate
        self.rejection_rate = rejection_rate
        self.denoise = denoise
        self.denoise_rate = denoise_rate
        self.max_T = max_T

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        # assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.model_variance = np.append(self.posterior_variance[1], self.betas[1:])
        self.model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))

        self.mapping_func = None # implement in train main()
        self.add_mask_noise = False # TODO

        # presaved as tensor
        self.sqrt_alphas_cumprod = th.from_numpy(self.sqrt_alphas_cumprod).to(device=device)
        self.sqrt_one_minus_alphas_cumprod = th.from_numpy(self.sqrt_one_minus_alphas_cumprod).to(device=device)
        self.log_one_minus_alphas_cumprod = th.from_numpy(self.log_one_minus_alphas_cumprod).to(device=device)
        self.sqrt_recip_alphas_cumprod = th.from_numpy(self.sqrt_recip_alphas_cumprod).to(device=device)
        self.sqrt_recipm1_alphas_cumprod = th.from_numpy(self.sqrt_recipm1_alphas_cumprod).to(device=device)
        self.posterior_variance = th.from_numpy(self.posterior_variance).to(device=device)
        self.posterior_log_variance_clipped = th.from_numpy(self.posterior_log_variance_clipped).to(device=device)
        self.posterior_mean_coef1 = th.from_numpy(self.posterior_mean_coef1).to(device=device)
        self.posterior_mean_coef2 = th.from_numpy(self.posterior_mean_coef2).to(device=device)
        self.model_log_variance = th.from_numpy(self.model_log_variance).to(device=device)
        self.model_variance = th.from_numpy(self.model_variance).to(device=device)

        ### I modified ###
        self._lambda = _lambda
        self.argsort_idx = freq_rank()
        self.batch_sum_appear = 0
        self.batch_num = 0
        self.temp = th.empty((2000, 0), device='cuda')
        self.temp2 = th.empty((2000, 0), device='cuda')
        self.temp3 = []
        self.batch_cnt = 1
        ##################

    def training_losses(self, model, *args, **kwargs):
        self.model = model
        return self.training_losses_seq2seq(model, *args, **kwargs)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None, mask=None, mean_embed=None, importance_score=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        
        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape

        if mask != None:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)

        if mean_embed is None:
            x_t = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
            )
        else:
            x_t = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * (x_start - mean_embed[None, None])
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise + mean_embed[None, None] * mask
            )

        if self.denoise:
            ### I modified ###
            # TODO: get noisy schedual 
            if importance_score is None:
                mask_rate = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape[:2]) * self.denoise_rate
            else:
                mask_rate = calculate_mask_rate(importance_score, t, self.num_timesteps, self._lambda)
            ##################
            # mask_rate = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape[:2]) * self.denoise_rate
            # print(mask_rate)
            # print('mask_rate', mask_rate.shape)
            # print((mask_rate_1-mask_rate).mean())
            # exit()
            random_mask = mask_rate.bernoulli()[..., None]
            random_mask = random_mask.expand(x_start.shape)
            mean_embed_expand = mean_embed[None, None].expand(x_start.shape)
            x_t = th.where(random_mask==0, x_t, mean_embed_expand)

        if mask == None:
            return x_t
        else:
            # mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return th.where(mask==0, x_start, x_t)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, last_tokens=None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B = x.size(0)
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        
        # for fixedlarge, we set the initial (log-)variance like so
        # to get a better decoder log likelihood.
        
        model_variance = _extract_into_tensor(self.model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(self.model_log_variance, t, x.shape)

        def process_xstart(x):
            current_tokens = None
            rejection_mask = None
            if denoised_fn is not None:
                raw_x = x
                x, current_tokens = denoised_fn(x, t)
                rejection_mask = None
                if self.rejection_rate > 0:
                    if last_tokens is not None:
                        rejection_rate = last_tokens.eq(current_tokens).float() * self.rejection_rate * t[:, None] / self.max_T
                        rejection_mask = rejection_rate.bernoulli()[..., None]
                        x = raw_x * rejection_mask + x * (1 - rejection_mask)
                else:
                    current_tokens = None
            if clip_denoised:
                return x.clamp(-1, 1)
            return x, current_tokens, rejection_mask

        if self.predict_xstart:
            pred_xstart, pred_xstart_tokens, rejection_mask = process_xstart(model_output)
        else:
            ### model is used to predict eps
            pred_xstart, pred_xstart_tokens, rejection_mask = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "last_tokens": pred_xstart_tokens,
            "rejection_mask": rejection_mask,
        }

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
            top_p=None, mask=None, x_start=None, last_tokens=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            last_tokens=last_tokens,
        )
        if top_p is not None and top_p > 0:
            noise = th.randn_like(x)
            replace_mask = th.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (th.abs(noise) <= top_p).all()

        else:
            noise = th.randn_like(x)
        
        if out['rejection_mask'] is not None:
            noise = noise * (1 - out['rejection_mask'])

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        if self.denoise:
            # mask_rate = nonzero_mask * th.exp(0.5 * out["log_variance"]) * self.denoise_rate
            mask_rate = calculate_mask_rate(
                th.zeros(sample.shape[:2], device=sample.device), 
                t, 
                self.num_timesteps, 
                0
            ) # I modified

            # print('p_sample in solver?', mask_rate.shape) # NO
            # random_mask = mask_rate[:,:,0].bernoulli()[..., None] # orgin
            random_mask = mask_rate.bernoulli()[..., None] # I modified
            # random_mask = mask_rate.bernoulli() # I modified, with masked record
            # random_mask = mask_rate[:,:,0].bernoulli() # another try

            # print(random_mask.shape, self.masked_record.shape)
            # self.masked_record = th.logical_and(random_mask, self.masked_record) # I modified
            # random_mask = self.masked_record[..., None]
            random_mask = random_mask.expand(x_start.shape)
            
            mean_embed = model.mean_embed
            mean_embed_expand = mean_embed[None, None].expand(x_start.shape)
            sample_without_mask = sample + 0
            sample = th.where(random_mask==0, sample, mean_embed_expand)

        if mask == None:
            pass
        else:
            sample = th.where(mask==0, x_start, sample)

        return {
            "sample": sample, 
            "pred_xstart": out["pred_xstart"],
            "greedy_mean": out["mean"], 
            "out": out,
            "last_tokens": out["last_tokens"],
            "masked_xstart": th.where(random_mask==0, out["pred_xstart"], mean_embed_expand),
            "sample_without_mask": sample_without_mask
        }

    
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        collect_every_step=False,
        sth=True,
        **kwargs
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param clamp_step: in clamp_first mode, choose end clamp step, otherwise starting clamp step
        :param clamp_first: bool, clamp_first mode
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """

        word_freq = th.load(f'./word_freq/bert-base-uncased_qqp_nocount_special.pt').cuda()

        final = {'x0':[], 'xt':[], 'sample_without_mask':[]} if collect_every_step else []

        # model.lm_head.weight[103] = model.mean_embed # same meaning as next line
        # model.word_embedding.weight[103] = model.mean_embed

        # check_EF = True
        check_EF = False

        p_sample_loop_progressive_result = self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            top_p=top_p,
            clamp_step=clamp_step,
            clamp_first=clamp_first,
            mask=mask,
            x_start=x_start
        )

        sample_to_feed_into_criterion = []
        dists = []
        freqs = []
        t1, t2 = 2, self.max_T-2

        from tqdm import tqdm
        for T_t, sample in enumerate(tqdm(p_sample_loop_progressive_result, total=self.max_T, position=1, leave=False), start=1):
            if collect_every_step:
                final['x0'].append(sample['masked_xstart'].cpu())
                final['xt'].append(sample['sample'])
                final['sample_without_mask'].append(sample['sample_without_mask'])

            else:
                final = sample['sample']
                
                # collect sample to feed into criterion, for every t
                if check_EF:
                    self.temp3.append(final)
                    with th.no_grad():
                        # logits = model.get_logits(final, 0)#.to('cpu') # bsz, seqlen, vocab
                        # cands_indices = th.topk(logits, k=1, dim=-1).indices
                        # sample_to_feed_into_criterion.append(cands_indices)

                        model.word_embedding.weight[103, :] = model.mean_embed
                        logits = model.get_logits(final, 0)#.to('cpu') # bsz, seqlen, vocab
                        argsort_idx = self.argsort_idx.tolist()
                        argsort_idx.pop(103)
                        argsort_idx = [103] + argsort_idx
                        unlock_bound = (len(argsort_idx)*T_t) // self.max_T
                        logits[..., argsort_idx[:unlock_bound]] = - float('inf')
                        cands_indices = th.topk(logits, k=1, dim=-1).indices

                        non_MASK = (cands_indices != 103).squeeze()
                        non_MASK_sum_dist = ((final - model.mean_embed).norm(dim=-1) * non_MASK).sum(dim=(0, 1,)) 
                        non_MASK_sum = non_MASK.sum(dim=(0, 1,))
                        
                        # dist = (final - model.mean_embed).norm(dim=-1).mean()
                        dist = non_MASK_sum_dist / non_MASK_sum
                        # print(word_freq[cands_indices.cpu()].shape, non_MASK.shape)
                        non_MASK_sum_appear = (word_freq[cands_indices].squeeze() * non_MASK).sum(dim=(0, 1,)) 
                        freq = non_MASK_sum_appear / non_MASK_sum

                        # print(dist)
                        # print(freq)

                        # 可以連batchsize的平均都算 只要注意所有batch都一樣大
                        dists.append(dist)
                        freqs.append(freq)

        if check_EF:
            s = th.stack(self.temp3)
            th.save(self.temp3, f'./decode_embs/batch{self.batch_cnt}.pt')
            self.batch_cnt += 1
            self.temp3 = []

            # print(dists)
            # print(freqs)
            dists = th.stack(dists)
            freqs = th.stack(freqs)
            self.temp = th.cat((self.temp, dists[..., None]), dim=-1)
            self.temp2 = th.cat((self.temp2, freqs[..., None]), dim=-1)
            th.save(self.temp.mean(dim=-1), 'EF_evid_dist0.pt')
            th.save(self.temp2.mean(dim=-1), 'EF_evid_freq0.pt')

        return final

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None: # custom your the start point of x_0
            sample_x = noise
        else:
            model_log_variance = _extract_into_tensor(self.model_log_variance, th.tensor([self.num_timesteps - 1] * shape[0], device=device), shape)
            sample_x = th.randn(*shape, device=device) * th.exp(0.5 * model_log_variance)
        if model.mean_embed is not None:
            sample_x += model.mean_embed[None, None] * mask
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        last_tokens = None

        # I modified
        # initial masked record all masked, 1: masked, 0: non masked
        # self.masked_record = th.ones(sample_x.shape[:2], device=sample_x.device)

        for i in indices: # from T to 0
            t = th.tensor([i] * shape[0], device=device)
            if not clamp_first:
                if i > clamp_step:
                    denoised_fn_cur = None
                else:
                    denoised_fn_cur = denoised_fn
            else:
                if i >= clamp_step:
                    denoised_fn_cur = denoised_fn
                else:
                    denoised_fn_cur = None
            with th.no_grad():
                out = self.p_sample(
                    model,
                    sample_x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn_cur,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                    mask=mask,
                    x_start=x_start,
                    last_tokens=last_tokens,
                )
                yield out
                sample_x = out["sample"]
                last_tokens = out["last_tokens"]


    def _get_x_start(self, x_start_mean, std):
        '''
        Word embedding projection from {Emb(w)} to {x_0}
        :param x_start_mean: word embedding
        :return: x_0
        '''
        noise = th.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        return (
             x_start_mean + std * noise
        )

    def _token_discrete_loss(self, x_t, get_logits, input_ids, mask=None, truncate=False, t=None, temperature=1):
        '''
        the loss of -log p(w|z_0)
        :param x_start_mean: word embedding
        :return: x_0
        '''
        reshaped_x_t = x_t
        logits = get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        loss_fct = th.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1))/temperature, input_ids.view(-1)).view(input_ids.shape)
        if mask != None:
            decoder_nll *= mask
        if mask != None:
            decoder_nll = decoder_nll.sum(dim=-1)/mask.sum(dim=-1)
        else:
            decoder_nll = decoder_nll.mean(dim=-1)

        return decoder_nll

    def _x0_helper(self, model_output, x, t):

        if self.predict_xstart:
            pred_xstart = model_output
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else: # predict eps
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        return {'pred_xprev':pred_prev, 'pred_xstart':pred_xstart}

    def training_losses_seq2seq(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs. # not used unless fixing the input embeddings
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        assert 'input_ids' in model_kwargs
        
        input_ids_x = model_kwargs.pop('input_ids').to(t.device)
        input_ids_mask = model_kwargs.pop('input_mask').to(t.device)
        x_start_mean = model.model.module.get_embeds(input_ids_x)

        ### I modified ###
        # importance BERT should only attend on tgt to calculate importance
        non_padding = input_ids_x != 0
        target_mask = th.logical_and(input_ids_mask, non_padding)
        target_mask[(1-input_ids_mask).sum(dim=-1, keepdim=True)] = False

        # importance_score = importance(input_ids_x, target_mask, self.importance_estimate_model)
        entropy = H(input_ids_x.cpu(), target_mask)
        ##################
        
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   th.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)
        x_start = self._get_x_start(x_start_mean, std)
        if noise is None:
            noise = th.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise, mask=input_ids_mask, mean_embed=model.mean_embed, importance_score=entropy) # reparametrization trick.

        get_logits = model.model.module.get_logits

        terms = {}

        target = x_start
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        assert model_output.shape == target.shape == x_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)

        model_out_x_start = self._x0_helper(model_output, x_t, t)['pred_xstart'] # predicted_xstart = model_output
        t0_mask = (t == 0)
        t0_loss = mean_flat((x_start_mean - model_out_x_start) ** 2)
        terms["mse"] = th.where(t0_mask, t0_loss, terms["mse"])

        # I modified, add contrastive loss
        embedding_weight = model.model.module.word_embedding.weight[self.argsort_idx]
        rand_ga = th.randint(1, len(embedding_weight), size=(1,))[0]
        anchor = model.mean_embed.detach()
        pos = embedding_weight[:rand_ga]
        neg = embedding_weight[rand_ga:]
        terms["triplet"] = triplet_margin_loss(anchor, pos, neg, margin=1e-4)
        terms["emb norm"] = (embedding_weight - anchor[None]).norm(dim=-1).mean()

        # I modified, add KL approx regurization
        posterior_mean, posterior_variance, log_posterior_variance = self.q_posterior_mean_variance(x_start, x_t, t)
        reverse_mean, reverse_variance, log_reverse_variance = self.q_posterior_mean_variance(model_out_x_start, x_t, t)
        q_mean = th.stack([model.mean_embed[None, None].expand(posterior_mean.shape), posterior_mean], dim=-2)
        p_mean = th.stack([model.mean_embed[None, None].expand(reverse_mean.shape), reverse_mean], dim=-2)
        small_var_shape = posterior_variance[:, :, 0].shape
        small_var = th.full(small_var_shape, 1e-8, device=posterior_variance.device)
        log_small_var = th.log(small_var)
        q_var = th.stack([small_var, posterior_variance[:, :, 0]], dim=-1)
        p_var = th.stack([small_var, reverse_variance[:, :, 0]], dim=-1)
        q_log_var = th.stack([log_small_var, log_posterior_variance[:, :, 0]], dim=-1)
        p_log_var = th.stack([log_small_var, log_reverse_variance[:, :, 0]], dim=-1)
        b = t / self.max_T
        q_weight = th.stack([b, 1-b], dim=-1)[:, None].expand(q_var.shape)
        p_weight = q_weight
        q = _gaussian_extractor(q_mean, q_var, q_weight, q_log_var)
        p = _gaussian_extractor(p_mean, p_var, p_weight, p_log_var)
        error_threshold = 1e-2
        d_prod = D_prod(q, p)
        d_var = D_var(q, p)
        approx_active = (d_var - d_prod) < error_threshold
        d_mean = ((d_var + d_prod)/2) * approx_active
        terms["d_mean"] = d_mean.mean(dim=1).squeeze()
        terms["d_act"] = approx_active.float().mean()

        out_mean, _, _ = self.q_mean_variance(x_start, th.LongTensor([self.num_timesteps - 1]).to(x_start.device))
        tT_loss =  mean_flat(out_mean ** 2)

        decoder_nll = self._token_discrete_loss(x_start, get_logits, input_ids_x, temperature=1) # embedding regularization
        terms["nll"] = self._token_discrete_loss(model_out_x_start, get_logits, input_ids_x, mask=input_ids_mask, truncate=True, t=t) # x_0->model_out_x_start

        terms["loss"] = terms["mse"] + decoder_nll + tT_loss + terms["triplet"] + 1e-2*terms["emb norm"] + terms["d_mean"]
        # terms["loss"] = terms["mse"] + decoder_nll + tT_loss + terms["d_mean"]
        # terms["loss"] = terms["mse"] + decoder_nll + tT_loss

        if model.mean_embed is not None:
            terms["loss"] = terms["loss"] + self.reg_rate * model.mean_embed.norm(p=2).sum()

        return terms

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if isinstance(arr, np.ndarray):
        res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    else:
        res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        # print(kwargs.keys())
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
    
    @property
    def mean_embed(self):
        from torch.nn.parallel.distributed import DistributedDataParallel as DDP
        from torch.nn.parallel.distributed import DistributedDataParallel as DDP_apex
        if isinstance(self.model, DDP) or isinstance(self.model, DDP_apex):
            return self.model.module.mean_embed
        else:
            return self.model.mean_embed
