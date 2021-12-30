import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def compute_grid(image_size, dtype=torch.float32, device='cpu'):

    dim = len(image_size)

    if(dim == 2):
        nx = image_size[0]
        ny = image_size[1]

        x = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)

        x = x.expand(ny, -1)
        y = y.expand(nx, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)

        return torch.cat((x, y), 3).to(dtype=dtype, device=device)

    elif(dim == 3):
        nz = image_size[0]
        ny = image_size[1]
        nx = image_size[2]

        x = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        z = torch.linspace(-1, 1, steps=nz).to(dtype=dtype)

        x = x.expand(ny, -1).expand(nz, -1, -1)
        y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
        z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(4)
        y.unsqueeze_(0).unsqueeze_(4)
        z.unsqueeze_(0).unsqueeze_(4)

        return torch.cat((x, y, z), 4).to(dtype=dtype, device=device)

def warp_images(images, displacement):
    dim = len(images[0].size())-2
    warpedIc = []
    for k in range(len(images)):
        if dim == 2:
            theta = param2theta(torch.tensor([[1, 0, 0], [0, 1, 0]], device=images[0].device).unsqueeze(0).float(),
                                displacement.size(2), displacement.size(3))
            id = F.affine_grid(theta, displacement[0, 0, :, :].squeeze().unsqueeze(0).unsqueeze(0).size(), align_corners=True)
            warpedIc.append(F.grid_sample(images[k], id + displacement[k, :, :, :].squeeze().permute(1, 2, 0).unsqueeze(0), align_corners=True))
        elif dim == 3:
            theta = param2theta3(torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                                              device=images[0].device).unsqueeze(0).float(), displacement.size(4), displacement.size(2), displacement.size(3))
            id = F.affine_grid(theta, displacement[0, 0, :, :, :].squeeze().unsqueeze(0).unsqueeze(0).size())
            warpedIc.append(F.grid_sample(images[k], id + displacement[k, :, :, :, :].squeeze().permute(1, 2, 3, 0).unsqueeze(0)))

    return warpedIc

def prolong_displacements(displacement, new_size):
    new_size = new_size[2:]
    dim = len(new_size)
    num_images = displacement.size()[0]

    prolonged_displacement = []
    for k in range(num_images):
        prolonged_tmp = [] ## save single coordinates for later concatenation
        for j in range(dim):
            if dim == 2:
                theta = param2theta(torch.tensor([[1, 0, 0], [0, 1, 0]],
                                                 device=displacement[0,...].device).unsqueeze(0).float(),
                                    new_size[0], new_size[1])
            elif dim == 3:
                theta = param2theta3(torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                                                  device=displacement[0,...].device).unsqueeze(0).float(),
                                     new_size[2], new_size[0], new_size[1])
            id = F.affine_grid(theta, ([1, dim] + new_size.tolist()), align_corners=True)
            prolonged_tmp.append(F.grid_sample(displacement[k,j,...].unsqueeze(0).unsqueeze(0), id, align_corners=True))
        prolonged_displacement.append(torch.stack(prolonged_tmp).squeeze())

    return torch.stack(prolonged_displacement)

def displacement2grid(displacement):
    dim = displacement.size()[1]
    num_images = displacement.size()[0]
    perm_a = np.roll(range(dim+1),-1).tolist() # permutation indices for grid dimensions
    perm_b = np.roll(range(dim+2),-1).tolist() # perm. indices for final output (purpose: number of images is last)

    grids = []
    for k in range(num_images):
        if dim == 2:
            theta = param2theta(torch.tensor([[1, 0, 0], [0, 1, 0]],
                                             device=displacement[0,...].device).unsqueeze(0).float(),
                                displacement.size(2), displacement.size(3))
        elif dim == 3:
            theta = param2theta3(torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                                              device=displacement[0,...].device).unsqueeze(0).float(),
                                 displacement.size(4), displacement.size(2), displacement.size(3))

        id = F.affine_grid(theta, ([1, dim] + torch.tensor(displacement.size()[2:]).tolist()), align_corners=True)
        grid_tmp = id[0,...] + displacement[k, ...].permute(perm_a).unsqueeze(0).unsqueeze(0)
        grids.append(grid_tmp.squeeze())

    return torch.stack(grids).permute(perm_b)

def param2theta(param, w, h):
    theta = torch.zeros([param.size(0), 2, 3]).to(param.device)
    theta[:, 0, 0] = param[:, 0, 0]
    theta[:, 0, 1] = param[:, 0, 1] * h / w
    theta[:, 0, 2] = param[:, 0, 2] * 2 / w + theta[:, 0, 0] + theta[:, 0, 1] - 1
    theta[:, 1, 0] = param[:, 1, 0] * w / h
    theta[:, 1, 1] = param[:, 1, 1]
    theta[:, 1, 2] = param[:, 1, 2] * 2 / h + theta[:, 1, 0] + theta[:, 1, 1] - 1

    return theta

def param2theta3(param, d, w, h):
    theta = torch.zeros([param.size(0), 3, 4]).to(param.device)
    theta[:, 0, 0] = param[:, 0, 0]
    theta[:, 0, 1] = param[:, 0, 1] * w / d
    theta[:, 0, 2] = param[:, 0, 2] * h / d
    theta[:, 0, 3] = param[:, 0, 0] + theta[:, 0, 2] + theta[:, 0, 1] + 2 / d * param[:, 0, 3] - 1
    theta[:, 1, 0] = param[:, 1, 0] * d / w
    theta[:, 1, 1] = param[:, 1, 1]
    theta[:, 1, 2] = param[:, 1, 2] * h / w
    theta[:, 1, 3] = param[:, 1, 1] + 2 / w * param[:, 1, 3] + theta[:, 1, 0] + theta[:, 1, 2] - 1
    theta[:, 2, 0] = param[:, 2, 0] * d / h
    theta[:, 2, 1] = param[:, 2, 1] * w / h
    theta[:, 2, 2] = param[:, 2, 2]
    theta[:, 2, 3] = param[:, 2, 3] * 2 / h + param[:, 2, 2] + theta[:, 2, 1] + theta[:, 2, 0] - 1

    return theta

def grad3d(img, h=None):

    """"
    Args:
        img:
        h (vec):
    """
    B, C, D, H, W = img.size()

    if h is None:
        h = torch.ones(B, 3)

    # compute gradients on staggered grid
    conv_D = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=[2, 1, 1], padding=[1, 0, 0], bias=False, groups=C)
    conv_H = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=[1, 2, 1], padding=[0, 1, 0], bias=False, groups=C)
    conv_W = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=[1, 1, 2], padding=[0, 0, 1], bias=False, groups=C)
    conv_D.weight.requires_grad = False
    conv_H.weight.requires_grad = False
    conv_W.weight.requires_grad = False

    conv_D.weight.data = torch.Tensor([1, -1]).view(1, 1, 2, 1, 1).repeat(C, 1, 1, 1, 1)
    conv_H.weight.data = torch.Tensor([1, -1]).view(1, 1, 1, 2, 1).repeat(C, 1, 1, 1, 1)
    conv_W.weight.data = torch.Tensor([1, -1]).view(1, 1, 1, 1, 2).repeat(C, 1, 1, 1, 1)

    if img.is_cuda:
        conv_D = conv_D.to(img.device)
        conv_H = conv_H.to(img.device)
        conv_W = conv_W.to(img.device)
        h = h.to(img.device)

    h_D = h[:, 0].view(-1, 1, 1, 1, 1)
    h_H = h[:, 1].view(-1, 1, 1, 1, 1)
    h_W = h[:, 2].view(-1, 1, 1, 1, 1)

    dD = conv_D(img)
    dH = conv_H(img)
    dW = conv_W(img)

    return dD / h_D, dH / h_H, dW / h_W


def grad2d(img, h=None):

    """"
    Args:
        img:
        h (vec):
    """

    B, C, H, W = img.size()

    if h is None:
        h = torch.ones(B, 2)

    # compute gradients on staggered grid
    conv_H = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=[2, 1], padding=[1, 0], bias=False, groups=C)
    conv_W = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=[1, 2], padding=[0, 1], bias=False, groups=C)
    conv_H.weight.requires_grad = False
    conv_W.weight.requires_grad = False

    conv_H.weight.data = torch.Tensor([1, -1]).view(1, 1, 2, 1).repeat(C, 1, 1, 1)
    conv_W.weight.data = torch.Tensor([1, -1]).view(1, 1, 1, 2).repeat(C, 1, 1, 1)

    if img.is_cuda:
        conv_H = conv_H.to(img.device)
        conv_W = conv_W.to(img.device)
        h = h.to(img.device)

    h_H = h[:, 0].view(-1, 1, 1, 1)
    h_W = h[:, 1].view(-1, 1, 1, 1)

    dH = conv_H(img)
    dW = conv_W(img)

    return dH / h_H, dW / h_W

def image_list2stack_numpy(image_list):

    num_images = len(image_list)
    num_pixels = np.prod(image_list[0].size().tolist())

    stack = np.zeros(num_pixels, num_images)
    for k in range(num_images):
        stack[:, k] = image_list[k].flatten('F')

    return stack

def image_list2numpy(image_list):

    image_size = np.array(image_list[0].size()[2:])
    num_images = len(image_list)

    images = np.zeros(np.append(image_size, num_images))
    for k in range(num_images):
        images[...,k] = image_list[k].squeeze()

    return images

def get_omega(m, h=1, invert=False, centershift=True):
    tmp = np.zeros((m.size,2))
    tmp[:,1] = h*m
    y = tmp[0,1]; x = tmp[1,1]
    tmp[0,1] = x; tmp[1,1] = y

    if invert:
        omega = tmp.reshape(2*m.size)
        omg = omega.copy()
        omega[0] = omg[2]
        omega[1] = omg[3]
        omega[2] = omg[0]
        omega[3] = omg[1]
    else:
        omega = tmp.reshape(2*m.size)

    if centershift:
        h = np.array([h])
        if len(h.shape) > 1:
            if len(m.shape) == 2:
                omega[0:2] = omega[0:2] + h[0] / 2
                omega[2:] = omega[2:] + h[1] / 2
            elif len(m.shape) == 3:
                omega[0:2] = omega[0:2] + h[0] / 2
                omega[2:4] = omega[2:4] + h[1] / 2
                omega[4:] = omega[4:] + h[2] / 2
        else:
            omega = omega + h/2

    return omega

def compute_circle(image_size, radius, center):
    x, y = np.ogrid[-center[0]:image_size[0] - center[0], -center[1]:image_size[1] - center[1]]
    mask = x * x + y * y <= radius * radius
    circ = np.zeros(image_size)
    circ[mask] = 255

    return circ

def compute_ball(image_size, radius, center):
    x, y, z = np.ogrid[-center[0]:image_size[0] - center[0], -center[1]:image_size[1] - center[1],
            -center[2]:image_size[2] - center[2]]
    mask = x * x + y * y + z * z <= radius * radius
    ball = np.zeros(image_size)
    ball[mask] = 255

    return ball