import torch.nn as nn
import torch

class HyperNetwork(nn.Module):
    def __init__(self, target_size, hyp_dim) -> None:
        super().__init__()

        self.hyp_dim = hyp_dim
        self.target_size = target_size

        self.r = 32

        self.hyp_weight = nn.Sequential(
            # nn.Linear(hyp_dim, hdim),
            # nn.SiLU(),
            nn.Linear(hyp_dim, 2*target_size*self.r),
        )

        self.hyp_bias = nn.Sequential(
            # nn.Linear(hyp_dim, hdim),
            # nn.SiLU(),
            nn.Linear(hyp_dim, target_size),
        )

        # print(self.hyp_weight)
        
    def forward(self, x):
        hw = self.hyp_weight(x).view((len(x), 2, self.target_size*self.r))
        hw_down = hw[:, 0].view((len(x), self.target_size, self.r))
        hw_up = hw[:, 1].view((len(x), self.r, self.target_size))
        return {'down': hw_down, 'up': hw_up}, self.hyp_bias(x)
    
class TargetLinear(nn.Module):
    def __init__(self, target_size, hyp_dim) -> None:
        super().__init__()

        self.hyper = HyperNetwork(target_size, hyp_dim)

    def forward(self, target_ipt, hyper_ipt, hyp_mask=1):
        self.weight, self.bias = self.hyper(hyper_ipt)
        lin_out = target_linear(target_ipt, self.weight['down']@self.weight['up'], self.bias)
        # print('lin_out', lin_out.dtype)
        # print('hyp_mask', hyp_mask.dtype)
        # print('(lin_out * hyp_mask)', (lin_out * hyp_mask).dtype)
        # print('(lin_out * hyp_mask)', (lin_out * hyp_mask).sum(dim=-2).dtype)
        return (lin_out * hyp_mask).sum(dim=-2)
    
# class Target(nn.Module):
#     def __init__(self, target_size, hyp_dim) -> None:
#         super().__init__()

#         self.lin = nn.Sequential([
#             nn.Linear(target_size, 2*target_size),
#             nn.SiLU(),
#             nn.Linear(2*target_size, target_size)
#         ])

#     def forward(self, target_ipt, hyper_weight):
        
#         return target_linear(target_ipt, self.weight, self.bias)


def target_linear(x, w, b):
    # print('-'*8+'target_linear')
    # print(x.shape)
    # print(w.shape)
    # print(b.shape)
    ret = (x[:, :, None, None] @ w).squeeze() + b
    # print(ret.shape)
    # print('='*8)
    return ret
