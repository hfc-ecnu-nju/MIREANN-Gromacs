# GMX_QEQ_REANN NEW FILE: fixed-species inference; upstream weights unchanged.
"""Exact fixed-species, nonperiodic REANN charge inference specialization.

Reuses every trained layer/parameter. Replaces CUDA nonzero/scalar-driven
neighbor/element selection with fixed all-pair tensors and distance masks.
The C++ adapter validates qeq_static_species and qeq_static_pbc at model load.
This module is inference-only; no energy or chain-rule forces are introduced.
"""
import math
import torch
from torch import nn


class ElementNet(nn.Module):
    def __init__(self, source, species):
        super().__init__()
        self.elemental_nets = nn.ModuleDict(dict(source.elemental_nets.named_children()))
        self.outputneuron = source.outputneuron
        self.register_buffer('indices', torch.argsort(species, stable=True))
        # Fixed gather positions, no CUDA nonzero() during inference.
        self.register_buffer('inverse', torch.argsort(self.indices))
        self.sizes = [int((species == i).sum()) for i in range(len(self.elemental_nets))]

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        ordered = density.index_select(0, self.indices)
        values = torch.jit.annotate(list[torch.Tensor], [])
        offset = 0
        for i, network in enumerate(self.elemental_nets.values()):
            size = self.sizes[i]
            if size > 0:
                values.append(network(ordered[offset:offset+size]))
            offset += size
        return torch.cat(values, dim=0).index_select(0, self.inverse)


class StaticCharge(nn.Module):
    def __init__(self, source, species):
        super().__init__()
        self.totcharge = float(source.totcharge)
        self.cutoff = float(source.neigh_list.cutoff)
        self.count = len(species)
        self.nipsin = int(source.density.nipsin[0])
        self.register_buffer('qeq_static_species', species.clone())
        self.register_buffer('qeq_static_pbc', torch.zeros(3, dtype=torch.int32))
        self.gausswidth = source.gausswidth
        self.hardness = source.hardness
        self.nnmod = ElementNet(source.nnmod, species)
        self.messages = nn.ModuleList([ElementNet(m, species) for m in source.density.ocmod.children()])
        self.register_buffer('params', source.density.params.detach().index_select(0, species).clone())
        self.register_buffer('rs', source.density.rs.detach().index_select(0, species).clone())
        self.register_buffer('inta', source.density.inta.detach().index_select(0, species).clone())
        self.register_buffer('hyper', source.density.hyper.detach().index_select(1, source.density.index_para).clone())
        pairs = [(i,j) for i in range(self.count) for j in range(self.count) if i != j]
        self.register_buffer('first', torch.tensor([i for i,j in pairs], dtype=torch.int64))
        self.register_buffer('second', torch.tensor([j for i,j in pairs], dtype=torch.int64))
        widths = self.gausswidth.detach().index_select(0, species)
        self.register_buffer('gamma', torch.sqrt(widths[:,None]**2 + widths[None,:]**2))
        self.register_buffer('diagonal', (self.hardness.detach().index_select(0,species) + widths.reciprocal()/math.sqrt(math.pi)).diag())
        self.register_buffer('eye', torch.eye(self.count, dtype=torch.float64))
        augmented_mask = torch.ones(self.count+1,self.count+1,dtype=torch.float64)
        augmented_mask[-1,-1] = 0.0
        self.register_buffer('augmented_mask',augmented_mask)
        # Preserve the supplied model's float32 sqrt(2.) scalar exactly.
        self.sqrt2 = float(torch.sqrt(torch.tensor(2.0)))

    def obtain(self, orbital: torch.Tensor, coeff: torch.Tensor, iteration: int) -> torch.Tensor:
        weighted = orbital * coeff.index_select(0,self.second)[:,None,:]
        summed = weighted.reshape(self.count,self.count-1,orbital.size(1),orbital.size(2)).sum(dim=1)
        projected = torch.einsum('ijk,jkm->ijm', [summed, self.hyper[iteration]])
        return projected.square().sum(dim=1)

    def forward(self, pbc: torch.Tensor, cart: torch.Tensor, cell: torch.Tensor,
                species: torch.Tensor, mass: torch.Tensor) -> tuple[torch.Tensor]:
        # Same whole-molecule wrapping and centering as original Neigh_List.
        # inv_ex avoids the unconditional CUDA error-check synchronization.
        inv_cell = torch.linalg.inv_ex(cell, check_errors=False)[0]
        fractional = cart @ inv_cell
        cart = (fractional - torch.round(fractional-fractional[0])) @ cell
        cart = cart - (mass[:,None]*cart).sum(dim=0)[None,:]/mass.sum()
        vector = cart.index_select(0,self.first) - cart.index_select(0,self.second)
        distance = torch.linalg.vector_norm(vector, dim=-1)
        active = (distance < self.cutoff) & (distance > 0.001)
        cutoff = (0.5*torch.cos(distance*(math.pi/self.cutoff))+0.5).square()*active
        angular = torch.jit.annotate(list[torch.Tensor], [cutoff[None,:]])
        power = cutoff[None,:]
        for _ in range(1,self.nipsin):
            power = torch.einsum('ji,ki->jki',[power,vector.t()]).reshape(-1,len(self.first))
            angular.append(power)
        radial = torch.exp(self.inta.index_select(0,self.second) *
                           (distance[:,None]-self.rs.index_select(0,self.second)).square())
        orbital = torch.cat(angular,dim=0).t()[:,:,None]*radial[:,None,:]
        coeff = self.params.clone()
        density = self.obtain(orbital,coeff,0)
        for i, message in enumerate(self.messages):
            coeff = coeff + message(density)
            density = self.obtain(orbital,coeff,i+1)
        chi = self.nnmod(density)
        distances = torch.linalg.vector_norm(cart[:,None,:]-cart[None,:,:],dim=-1)
        # Diagonal uses trained hardness. Adding eye keeps division finite.
        matrix = torch.erf(distances/self.sqrt2/self.gamma)/(distances+self.eye)
        matrix = matrix*(1-self.eye)+self.diagonal
        matrix = torch.nn.functional.pad(matrix,(0,1,0,1),value=1.0)
        matrix = matrix*self.augmented_mask
        rhs = torch.nn.functional.pad(chi,(0,0,0,1),value=self.totcharge).squeeze(-1)
        charge = torch.linalg.solve_ex(matrix,rhs,check_errors=False)[0]
        return (charge.detach(),)
