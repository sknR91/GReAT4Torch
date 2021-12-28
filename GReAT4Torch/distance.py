import torch
import torch.nn.functional as F
import numpy as np
from . import utils
from torch import nn


import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/kai/Documents/great')
import GReAT ## import numpy based GReAT for some helpful tools

class _Distance(torch.nn.modules.Module):
    def __init__(self, images, size_average=True, reduce=True):
        super(_Distance, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self.name = "parent"

        self._warped_images = None
        self._weight = 1

        self._images = images
        self._dtype = images[0].dtype
        self._device = images[0].device
        self._m = images[0].size()
        self._dim = len(self._m)-2
        self._h = torch.ones(1, self._dim)

        assert self._images != None

    def get_warped_images(self):
        return self._warped_images[0, 0, ...].detach().cpu()

    def set_distance_weight(self, weight):
        self._weight = weight

    def set_pixel_spacing(self, pixel_spacing):
        self._h = pixel_spacing

    def return_distance(self, tensor):
        if self._size_average and self._reduce:
            return tensor.mean() * self._weight
        if not self._size_average and self._reduce:
            return tensor.sum() * self._weight
        if not self.reduce:
            return tensor * self._weight

class SqN(_Distance):
    def __init__(self, images, size_average=True, reduce=True):
        super(SqN, self).__init__(images, size_average, reduce)

        self.name = "sqn"
        self.images = images
        self.warped_images = None
        self.edge = 1e-3
        self.normalize = 'local'
        self.q = 4
        self.pplt = GReAT.plot()

    def set_edge_parameter(self, edge):
        self.edge = edge

    def set_normalization_method(self, normalize):
        self.normalize = normalize

    def set_q_parameter(self, q):
        self.q = q

    def _normalized_gradients(self, Ic, edge=1e-3, h=None, normalize='local'):
        m = Ic[0].size()
        dim = len(m[2:])
        B = m[0]
        C = m[1]

        if dim == 2:
            conv_x = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=[2, 1], bias=False, groups=C)
            conv_y = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=[1, 2], bias=False, groups=C)
            conv_x.weight.requires_grad = False
            conv_y.weight.requires_grad = False

            conv_x.weight.data = torch.Tensor([0.5, 0.5]).view(1, 1, 2, 1).repeat(C, 1, 1, 1)
            conv_y.weight.data = torch.Tensor([0.5, 0.5]).view(1, 1, 1, 2).repeat(C, 1, 1, 1)
        elif dim == 3:
            conv_x = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=[2, 1, 1], bias=False, groups=C)
            conv_y = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=[1, 2, 1], bias=False, groups=C)
            conv_z = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=[1, 1, 2], bias=False, groups=C)
            conv_x.weight.requires_grad = False
            conv_y.weight.requires_grad = False
            conv_z.weight.requires_grad = False

            conv_x.weight.data = torch.Tensor([0.5, 0.5]).view(1, 1, 2, 1, 1).repeat(C, 1, 1, 1, 1)
            conv_y.weight.data = torch.Tensor([0.5, 0.5]).view(1, 1, 1, 2, 1).repeat(C, 1, 1, 1, 1)
            conv_z.weight.data = torch.Tensor([0.5, 0.5]).view(1, 1, 1, 1, 2).repeat(C, 1, 1, 1, 1)

        Ic_x = []; Ic_y = []; Ic_z = []
        for k in range(len(Ic)):
            if dim == 2:
                tmp_x, tmp_y = utils.grad2d(Ic[k], h)
            elif dim == 3:
                tmp_x, tmp_y, tmp_z = utils.grad3d(Ic[k], h)
                Ic_z.append(conv_z(tmp_z))
            Ic_x.append(conv_x(tmp_x))
            Ic_y.append(conv_y(tmp_y))

        normIc_x = []; normIc_y = []; normIc_z = []
        for k in range(len(Ic)):
            if normalize == 'local':
                if dim == 2:
                    prod = torch.sqrt(Ic_x[k] * Ic_x[k] + Ic_y[k] * Ic_y[k] + edge ** 2)
                elif dim == 3:
                    prod = torch.sqrt(Ic_x[k] * Ic_x[k] + Ic_y[k] * Ic_y[k] + Ic_z[k] * Ic_z[k] + edge ** 2)
            else:
                if dim == 2:
                    prod = torch.sqrt(torch.sum(Ic_x[k] * Ic_x[k] + Ic_y[k] * Ic_y[k]) + edge ** 2)
                elif dim == 3:
                    prod = torch.sqrt(torch.sum(Ic_x[k] * Ic_x[k] + Ic_y[k] * Ic_y[k] + Ic_z[k] * Ic_z[k]) + edge ** 2)
            normIc_x.append(Ic_x[k] / prod)
            normIc_y.append(Ic_y[k] / prod)
            if dim == 3:
                normIc_z.append(Ic_z[k] / prod)

        return normIc_x, normIc_y, normIc_z

    def _sqnorm(self, A, q, U=None, S=None, V=None):
        if S is None:
            (_, S, _) = torch.svd(A, compute_uv=True)
        else:
            (U, _, V) = torch.svd(A)

        val = torch.sum(S ** q) ** (1 / q)

        return (val, U, S, V)

    def forward(self, displacement):
        self.warped_images = utils.warp_images(images=self.images, displacement=displacement)

        f = plt.figure(1)
        f.clf()
        plt.subplot(231)
        plt.imshow(self.warped_images[0].cpu().squeeze().detach().numpy())
        plt.subplot(232)
        plt.imshow(self.warped_images[1].cpu().squeeze().detach().numpy())
        plt.subplot(233)
        plt.imshow(self.warped_images[2].cpu().squeeze().detach().numpy())
        plt.subplot(234)
        theta = utils.param2theta(torch.tensor([[1, 0, 0], [0, 1, 0]],
                                          device=self.images[0].device).unsqueeze(0).float(),
                                  displacement.size(2), displacement.size(3))
        id = F.affine_grid(theta, displacement[0, 0, :, :].squeeze().unsqueeze(0).unsqueeze(0).size())
        self.pplt.plotGrid(id[0,...].cpu().detach().numpy()+displacement[0,...].permute(1,2,0).cpu().detach().numpy())
        plt.subplot(235)
        self.pplt.plotGrid(id[0,...].cpu().detach().numpy() + displacement[1,...].permute(1, 2, 0).cpu().detach().numpy())
        plt.subplot(236)
        self.pplt.plotGrid(id[0,...].cpu().detach().numpy() + displacement[2,...].permute(1, 2, 0).cpu().detach().numpy())
        plt.pause(0.001)

        m = self._m
        dim = self._dim
        num_batches = m[0]
        num_channels = m[1]
        h = self._h
        q = self.q

        nP = self.images[0].numel()
        hd = torch.prod(h)

        Ic_x, Ic_y, Ic_z = self._normalized_gradients(self.warped_images, self.edge, self._h, self.normalize)
        Ic_n = []
        for k in range(len(Ic_x)):
            if dim == 2:
                Ic_n.append(torch.cat([Ic_x[k].view(num_batches, num_channels, -1),
                                       Ic_y[k].view(num_batches, num_channels, -1)], dim=2))
            elif dim == 3:
                Ic_n.append(torch.cat([Ic_x[k].view(num_batches, num_channels, -1),
                                       Ic_y[k].view(num_batches, num_channels, -1),
                                       Ic_z[k].view(num_batches, num_channels, -1)], dim=2))
        Ic_n = torch.stack(Ic_n)
        Ic_n = Ic_n.permute((1, 2, 3, 0))

        rc, _, _, _ = self._sqnorm(Ic_n[0, 0, :, :], q)
        #hd = hd * np.sqrt(nP)
        #Dc = hd * rc
        Dc = rc

        return self.return_distance(-Dc)