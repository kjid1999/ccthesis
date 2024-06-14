import torch

def int_prod(f, g):
    '''
    Calculate integral of product of each 2 gaussian.
    Both gaussian are with cov like sigma^2I
    '''
    cat_vars = torch.cat((f.vars, g.vars))
    sum_cov = cat_vars[None, ...] + f.vars[:, None]
    normalize_factor = 1 / (2*torch.pi * sum_cov) ** (k/2)

    diff_mean = f.means[:, None] - torch.cat((f.means, g.means))[None, ...] # 差負號不會有問題
    power = -1/2 /sum_cov * torch.norm(diff_mean, dim=-1)**2

    return normalize_factor * torch.exp(power)

def D_prod(f, g, min_value=1e-12):
    zt = int_prod(f, g)
    z, t = torch.tensor_split(zt, 2, dim=1)
    w_sum_z = f.weights @ z
    w_sum_t = g.weights @ t
    w_sum_t = torch.where(w_sum_t < min_value, min_value, w_sum_t)
    D = f.weights @ torch.log(w_sum_z / w_sum_t)
    return D

def _KL_of_2_single(f, g):
    k = len(f.means[0])

    cat_vars = torch.cat((f.vars, g.vars))
    trace = k * (f.vars[:, None] / cat_vars[None, ...])

    ln = k * torch.log(cat_vars[None, ...] / f.vars[:, None])

    diff_mean = torch.cat((f.means, g.means))[None, ...] - f.means[:, None] # 差負號不會有問題
    norm_square = torch.norm(diff_mean, dim=-1)**2

    return 0.5 * (trace - k + norm_square/cat_vars[None] + ln)

def D_var(f, g, min_value=1e-12):
    KLs = _KL_of_2_single(f, g)
    exp_KL = torch.exp(-KLs)
    exp_KL_ff, exp_KL_fg = torch.tensor_split(exp_KL, 2, dim=1)
    w_sum_ff = f.weights @ exp_KL_ff
    w_sum_fg = g.weights @ exp_KL_fg
    w_sum_fg = torch.where(w_sum_fg < min_value, min_value, w_sum_fg)
    return f.weights @ torch.log(w_sum_ff/w_sum_fg)

def D_MC(f, g, n=1e4, min_value=1e-12):
    n = int(n)
    x = f.samples(n)
    gx = g(x)
    gx = torch.where(gx < min_value, min_value, gx)
    return (torch.log(f(x)) - torch.log(gx)).mean()

def D_mean(f, g):
    return (D_prod(f, g) + D_var(f, g))/2