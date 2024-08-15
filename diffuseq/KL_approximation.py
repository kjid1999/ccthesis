import torch
from math import log
from collections import namedtuple

def use_float32(func):
    def wrapper(*args, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float):
            return func(*args, **kwargs)
    return wrapper

def use_float64(func):
    def wrapper(*args, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float64):
            return func(*args, **kwargs)
    return wrapper

def _gaussian_extractor(mean, var, weight, log_var):
    '''
    mean: (B, T, 2, D), batchsize x length x [[mask], [non_mask mean]]
    var: (B, T, 2), batchsize x length x [small number for mask var, noisy scaduale]
    weights: (B, T, 2), batchsize x length x [mask_rate , 1-mask_rate]
    '''
    var = torch.clamp(var, min=1e-8)
    return namedtuple('Gaussian_params', ['means', 'vars', 'weights', 'log_vars'])(mean, var, weight, log_var)

@use_float32
def log_int_prod(f, g):
    '''
    Calculate integral of product of each 2 gaussian.
    Both gaussian are with cov like sigma^2I
    '''
    k = f.means.shape[-1]
    cat_vars = torch.cat((f.vars, g.vars), dim=-1)
    sum_cov = cat_vars[..., None, :] + f.vars[..., None]
    # normalize_factor = 1 / ((2*torch.pi * sum_cov) ** (k/2) + 1e-8)
    log_normalize_factor = -(k/2)*torch.log(2*torch.pi * sum_cov)

    diff_mean = f.means[..., None, :] - torch.cat((f.means, g.means), dim=-2)[..., None, :, :] # 差負號不會有問題
    power = -1/2 /sum_cov * torch.norm(diff_mean, dim=-1)**2

    # return torch.exp(log_normalize_factor + power)
    return log_normalize_factor + power

@use_float32
def D_prod(f, g):
    log_zt = log_int_prod(f, g)
    log_z, log_t = torch.tensor_split(log_zt, 2, dim=-1)
    log_w_z = torch.log(f.weights[..., None]) + log_z
    log_w_t = torch.log(g.weights[..., None]) + log_t
    diff_logsum = torch.logsumexp(log_w_z, dim=-2) - torch.logsumexp(log_w_t, dim=-2)

    D = f.weights[..., None, :] @ diff_logsum.squeeze(dim=-2)[..., None]
    # D = torch.nan_to_num(D, posinf=1e8, neginf=-1e8)
    return D.squeeze(dim=(-1))

@use_float32
def _KL_of_2_single(f, g):
    k = f.means.shape[-1]

    cat_vars = torch.cat((f.vars, g.vars), dim=-1)
    cat_log_vars = torch.cat((f.log_vars, g.log_vars), dim=-1)
    log_trace = log(k) + f.log_vars[..., None] -  cat_log_vars[..., None, :]

    ln = k * (cat_log_vars[..., None, :] - f.log_vars[..., None])

    diff_mean = f.means[..., None, :] - torch.cat((f.means, g.means), dim=-2)[..., None, :, :] # 差負號不會有問題
    norm_square = torch.norm(diff_mean, dim=-1)**2

    return 0.5 * (torch.exp(log_trace) - k + norm_square/cat_vars[..., None, :] + ln)

@use_float32
def D_var(f, g):
    KLs = _KL_of_2_single(f, g)
    exp_KL = torch.exp(-KLs)
    exp_KL_ff, exp_KL_fg = torch.tensor_split(exp_KL, 2, dim=-1)

    KL_ff, KL_fg = torch.tensor_split(KLs, 2, dim=-1)
    log_w_KL_ff = torch.log(f.weights[..., None]) - KL_ff
    log_w_KL_fg = torch.log(g.weights[..., None]) - KL_fg
    diff_logsum = torch.logsumexp(log_w_KL_ff, dim=-2) - torch.logsumexp(log_w_KL_fg, dim=-2)
    
    D =  f.weights[..., None, :] @ diff_logsum.squeeze(dim=-2)[..., None]
    return D.squeeze(dim=(-1))

def D_MC(f, g, n=1e4, min_value=1e-12):
    n = int(n)
    x = f.samples(n)
    gx = g(x)
    gx = torch.where(gx < min_value, min_value, gx)
    return (torch.log(f(x)) - torch.log(gx)).mean()

def D_mean(f, g):
    return (D_prod(f, g) + D_var(f, g))/2