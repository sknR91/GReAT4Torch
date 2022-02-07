import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class _Regularizer(torch.nn.modules.Module):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(_Regularizer, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self._alpha = 1
        self._dim = pixel_spacing.numel()
        self._h = pixel_spacing
        self.name = "parent"

    def set_alpha(self, alpha):
        self._alpha = alpha

    def get_alpha(self):
        return self._alpha

    def return_regularizer(self, tensor):
        if self._size_average and self._reduce:
            return self._alpha * tensor.mean()
        if not self._size_average and self._reduce:
            return self._alpha * tensor.sum()
        if not self._reduce:
            return self._alpha * tensor


class CurvatureRegularizer(_Regularizer):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(CurvatureRegularizer, self).__init__(pixel_spacing, size_average, reduce)

        self.name = "curvature"

        if self._dim == 2:
            self._regularizer = self._curvature_regularizer_2d
            self._displacement_normalization = self._displacement_normalization_2d
        elif self._dim == 3:
            self._regularizer = self._curvature_regularizer_3d
            self._displacement_normalization = self._displacement_normalization_3d

    def _curvature_regularizer_2d(self, displacement):

        # pixel spacing
        h = self._h

        # displacement needs to be normalized normalized (check function "displacementNormalization")
        # displacement = self._displacement_denormalization(displacement)
        h_big = h.view(-1, 2, 1, 1)
        if displacement.is_cuda:
            h_big = h_big.cuda()
        displacement *= h_big

        if displacement.is_cuda and not h.is_cuda:
            h = h.cuda()

        second_order_derivative = SecondOrderDerivative(h)
        L_x, L_y = second_order_derivative.forward(displacement)

        L = L_x + L_y
        L = L ** 2
        L = torch.sum(L, dim=[1, 2, 3])

        L = L * h.prod(dim=1)

        return L

    def _curvature_regularizer_3d(self, displacement):

        # pixel spacing
        h = self._h

        # displacement needs to be normalized normalized (check function "displacementNormalization")
        # displacement = self._displacement_denormalization(displacement)
        h_big = h_big = h.view(-1, 3, 1, 1, 1)
        if displacement.is_cuda:
            h_big = h_big.cuda()
        displacement *= h_big

        if displacement.is_cuda and not h.is_cuda:
            h = h.cuda()

        second_order_derivative = SecondOrderDerivative(h)
        L_x, L_y, L_z = second_order_derivative.forward(displacement)

        L = L_x + L_y + L_z
        L = L ** 2
        L = torch.sum(L, dim=[1, 2, 3, 4])

        L *= h.prod(dim=1)

        return L

    def _displacement_denormalization_2d(self, displacement):
        m = displacement.size()
        dim = self._dim

        scale = torch.ones(displacement.size())
        scale[:, 0, :, :] = scale[:, 0, :, :] * (m[2] - 1) / 2
        scale[:, 1, :, :] = scale[:, 1, :, :] * (m[3] - 1) / 2

        if displacement.is_cuda:
            return displacement * scale.cuda()
        else:
            return displacement * scale

    def _displacement_denormalization_3d(self, displacement):
        m = displacement.size()
        dim = self._dim

        scale = torch.ones(displacement.size())
        scale[:, 0, :, :, :] = scale[:, 0, :, :, :] * (m[2] - 1) / 2
        scale[:, 1, :, :, :] = scale[:, 1, :, :, :] * (m[3] - 1) / 2
        scale[:, 2, :, :, :] = scale[:, 2, :, :, :] * (m[3] - 1) / 2

        if displacement.is_cuda:
            return displacement * scale.cuda()
        else:
            return displacement * scale

    def _displacement_normalization_2d(self, displacement):
        m = displacement.size()
        dim = self._dim

        scale = torch.ones(displacement.size())
        scale[:, 0, :, :] = scale[:, 0, :, :] / (m[2] - 1) / 2
        scale[:, 1, :, :] = scale[:, 1, :, :] / (m[3] - 1) / 2

        if displacement.is_cuda:
            return displacement * scale.cuda()
        else:
            return displacement * scale

    def _displacement_normalization_3d(self, displacement):
        m = displacement.size()
        dim = self._dim

        scale = torch.ones(displacement.size())
        scale[:, 0, :, :, :] = scale[:, 0, :, :, :] / (m[2] - 1) / 2
        scale[:, 1, :, :, :] = scale[:, 1, :, :, :] / (m[3] - 1) / 2
        scale[:, 2, :, :, :] = scale[:, 2, :, :, :] / (m[4] - 1) / 2

        if displacement.is_cuda:
            return displacement * scale.cuda()
        else:
            return displacement * scale

    def forward(self, displacement):
        # set the supgradient to zeros
        value = self._regularizer(self._displacement_normalization(displacement))
        mask = value > 0
        value[mask] = torch.sqrt(value[mask])

        return self.return_regularizer(value)


class SecondOrderDerivative(_Regularizer):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(SecondOrderDerivative, self).__init__(pixel_spacing, size_average, reduce)

        self.name = "secondOrderDerivative"
        self._dim = pixel_spacing.numel()
        self._h = pixel_spacing
        self._padding_mode = 'zeros'

        if self._dim == 2:
            self._second_order_derivative = self._second_order_derivative_2d
        elif self._dim == 3:
            self._second_order_derivative = self._second_order_derivative_3d

    def set_padding_mode(self, padding_mode):
        assert padding_mode in ('zeros', 'reflect')
        self._padding_mode = padding_mode

    def _second_order_derivative_2d(self, tensor):
        m = tensor.size()
        dim = self._dim
        h = self._h
        padding_mode = self._padding_mode
        num_batches = m[0]
        num_channels = m[1]

        if padding_mode == 'reflect':
            img_x = F.pad(tensor, (0, 0, 1, 1), mode='constant', value=0)
            img_x[:, 0] = tensor[:, 1]
            img_x[:, -1] = tensor[:, -2]

            img_y = F.pad(tensor, (1, 1, 0, 0), mode='constant', value=0)
            img_y[:, :, 0] = tensor[:, :, 1]
            img_y[:, :, -1] = tensor[:, :, -2]

            conv_x = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=[3, 1], padding=[0, 0],
                               bias=False,
                               groups=num_channels)
            conv_y = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=[1, 3], padding=[0, 0],
                               bias=False,
                               groups=num_channels)
        elif padding_mode == 'zeros':
            conv_x = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=[2, 1], padding=[1, 0],
                               bias=False,
                               groups=num_channels)
            conv_y = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=[1, 2], padding=[0, 1],
                               bias=False,
                               groups=num_channels)

        conv_x.weight.requires_grad = False
        conv_y.weight.requires_grad = False
        conv_x.weight.data = torch.Tensor([1, -2, 1]).view(1, 1, 3, 1).repeat(num_channels, 1, 1, 1)
        conv_y.weight.data = torch.Tensor([1, -2, 1]).view(1, 1, 1, 3).repeat(num_channels, 1, 1, 1)

        if tensor.is_cuda:
            conv_x = conv_x.cuda()
            conv_y = conv_y.cuda()

        h_x = h[:, 0].view(-1, 1, 1, 1)
        h_y = h[:, 1].view(-1, 1, 1, 1)

        if padding_mode == 'reflect':
            L_x = conv_x(img_x)
            L_y = conv_y(img_y)
        elif padding_mode == 'zeros':
            L_x = conv_x(tensor)
            L_y = conv_y(tensor)

        return L_x / (h_x ** 2), L_y / (h_y ** 2)

    def _second_order_derivative_3d(self, tensor):
        m = tensor.size()
        dim = self._dim
        h = self._h
        padding_mode = self._padding_mode
        num_batches = m[0]
        num_channels = m[1]

        if padding_mode == 'reflect':
            img_x = F.pad(tensor, (0, 0, 0, 0, 1, 1), mode='constant', value=0)
            img_x[:, :, 0] = tensor[:, :, 1]
            img_x[:, :, -1] = tensor[:, :, -2]

            img_y = F.pad(tensor, (0, 0, 1, 1, 0, 0), mode='constant', value=0)
            img_y[:, :, :, 0] = tensor[:, :, :, 1]
            img_y[:, :, :, -1] = tensor[:, :, :, -2]

            img_z = F.pad(tensor, (1, 1, 0, 0, 0, 0), mode='constant', value=0)
            img_z[:, :, :, :, 0] = tensor[:, :, :, :, 1]
            img_z[:, :, :, :, -1] = tensor[:, :, :, :, -2]

            conv_x = nn.Conv3d(in_channels=num_channels, out_channels=num_channels, kernel_size=[3, 1, 1], padding=[0, 0, 0], bias=False,
                                   groups=num_channels)
            conv_y = nn.Conv3d(in_channels=num_channels, out_channels=num_channels, kernel_size=[1, 3, 1], padding=[0, 0, 0], bias=False,
                                   groups=num_channels)
            conv_z = nn.Conv3d(in_channels=num_channels, out_channels=num_channels, kernel_size=[1, 1, 3], padding=[0, 0, 0], bias=False,
                                   groups=num_channels)

        elif padding_mode == 'zeros':
            conv_x = nn.Conv3d(in_channels=num_channels, out_channels=num_channels, kernel_size=[3, 1, 1], padding=[1, 0, 0], bias=False,
                                   groups=num_channels)
            conv_y = nn.Conv3d(in_channels=num_channels, out_channels=num_channels, kernel_size=[1, 3, 1], padding=[0, 1, 0], bias=False,
                                   groups=num_channels)
            conv_z = nn.Conv3d(in_channels=num_channels, out_channels=num_channels, kernel_size=[1, 1, 3], padding=[0, 0, 1], bias=False,
                                   groups=num_channels)

        conv_x.weight.requires_grad = False
        conv_y.weight.requires_grad = False
        conv_z.weight.requires_grad = False
        conv_x.weight.data = torch.Tensor([1, -2, 1]).view(1, 1, 3, 1, 1).repeat(num_channels, 1, 1, 1, 1)
        conv_y.weight.data = torch.Tensor([1, -2, 1]).view(1, 1, 1, 3, 1).repeat(num_channels, 1, 1, 1, 1)
        conv_z.weight.data = torch.Tensor([1, -2, 1]).view(1, 1, 1, 1, 3).repeat(num_channels, 1, 1, 1, 1)

        if tensor.is_cuda:
            conv_x = conv_x.cuda()
            conv_y = conv_y.cuda()
            conv_z = conv_z.cuda()

        h_x = h[:, 0].view(-1, 1, 1, 1, 1)
        h_y = h[:, 1].view(-1, 1, 1, 1, 1)
        h_z = h[:, 2].view(-1, 1, 1, 1, 1)

        if padding_mode == 'reflect':
            L_x = conv_x(img_x)
            L_y = conv_y(img_y)
            L_z = conv_z(img_z)
        elif padding_mode == 'zeros':
            L_x = conv_x(tensor)
            L_y = conv_y(tensor)
            L_z = conv_z(tensor)

        return L_x / (h_x ** 2), L_y / (h_y ** 2), L_z / (h_z ** 2)

    def forward(self, tensor):
        return self._second_order_derivative(tensor)
