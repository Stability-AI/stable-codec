import torch
import torch.nn as nn

from typing import List, Tuple
from einops import rearrange
from .fsq import DitheredFSQ

class ResidualFSQBottleneck(nn.Module):
    def __init__(self, stages: List[Tuple[List[int], float]]):
        super().__init__()

        # 1st for single_tokens, others - residuals.
        self.quantizers = nn.ModuleList([
            DitheredFSQ(levels=levels, scale=scale).eval().requires_grad_(False)
            for (levels, scale) in stages])

        self.n_codebooks = len(stages)
        self.codebook_size = sum(map(len, stages)) * self.n_codebooks

    def encode(self, x):
        input_dtype = x.dtype
        z = torch.tanh(x.to(torch.float64))
        z = rearrange(z, "b c n -> b n c")

        r = z
        res_ids = []
        for quantizer in self.quantizers:
            q, ids = quantizer(r, skip_tanh=True)
            r = r - q.to(torch.float64)
            res_ids.append(ids)

        return res_ids

    def decode(self, res_ids):
        z = sum([
            q.indices_to_codes(res_ids[i])
            for (i, q) in enumerate(self.quantizers)
        ])
        return rearrange(z, "b n c -> b c n")

if __name__ == "__main__":
    fsq = DitheredFSQ([17, 17, 17, 17, 17, 17]).eval().requires_grad_(False)
    # res_fsq = ResidualFSQBottleneck(stages=[
    #     ([5, 5, 5, 5, 5, 5], 1.0),
    #     ([5, 5, 5, 5, 5, 5], 0.25),
    # ]).eval().requires_grad_(False)
    res_fsq = ResidualFSQBottleneck(stages=[
        ([3, 3, 3, 3, 3, 3], 1.0),
        ([3, 3, 3, 3, 3, 3], 0.5),
        ([3, 3, 3, 3, 3, 3], 0.25),
        ([3, 3, 3, 3, 3, 3], 0.125),
    ]).eval().requires_grad_(False)

    x = torch.rand(1, 6, 1)

    z1 = res_fsq.decode(res_fsq.encode(x))

    _, y2 = fsq(rearrange(x, "b c n -> b n c"))
    z2 = rearrange(fsq.indices_to_codes(y2), "b n c -> b c n")

    print(z1)
    print(z2)
    assert (z1 == z2).all()
