import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
import GReAT4Torch
import nibabel as nib
from scipy.ndimage import measurements
import inspect
import datetime
import json
import Image
import pydicom as dicom
import platform
import pathlib
import tkinter as tk
from tkinter import filedialog

warnings.filterwarnings("ignore")


def compute_grid(image_size, dtype=torch.float32, device=torch.device('cpu')):
    m = torch.tensor(image_size[2:])
    dim = m.numel()
    perm = np.roll(range(dim + 1), 1).tolist()

    if dim == 2:
        theta = param2theta(torch.tensor([[1, 0, 0], [0, 1, 0]], device=device).unsqueeze(0).float(), m[0], m[1])
    elif dim == 3:
        theta = param2theta3(torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                                          device=device).unsqueeze(0).float(), m[2], m[0], m[1])
    grid = F.affine_grid(theta, ([1, dim] + m.tolist()), align_corners=True).squeeze()

    return grid.permute(perm).unsqueeze(0)


def compute_covariance_matrix(x, I):

    if len(I.size()) > 1:
        I = I.view(-1)
    dim = x.size(1)

    if dim == 2:
        variance = torch.tensor([[torch.sum(x[:, 0] * x[:, 0] * I), torch.sum(x[:, 0] * x[:, 1] * I)],
                                [torch.sum(x[:, 1] * x[:, 0] * I), torch.sum(x[:, 1] * x[:, 1] * I)]])
    elif dim == 3:
        variance = torch.tensor([[torch.sum(x[:, 0] * x[:, 0] * I), torch.sum(x[:, 0] * x[:, 1] * I), torch.sum(x[:, 0] * x[:, 2] * I)],
                                 [torch.sum(x[:, 1] * x[:, 0] * I), torch.sum(x[:, 1] * x[:, 1] * I), torch.sum(x[:, 1] * x[:, 2] * I)],
                                 [torch.sum(x[:, 2] * x[:, 0] * I), torch.sum(x[:, 2] * x[:, 1] * I), torch.sum(x[:, 2] * x[:, 2] * I)]])
    sum_I = torch.sum(I)

    return torch.divide(variance, sum_I)


def warp_images(images, displacement):
    dim = len(images[0].size()) - 2
    warpedIc = []
    for k in range(len(images)):
        if dim == 2:
            theta = param2theta(torch.tensor([[1, 0, 0], [0, 1, 0]], device=images[0].device).unsqueeze(0).float(),
                                displacement.size(2), displacement.size(3))
            id = F.affine_grid(theta, displacement[0, 0, :, :].squeeze().unsqueeze(0).unsqueeze(0).size(),
                               align_corners=True)
            warpedIc.append(
                F.grid_sample(images[k], id + displacement[k, :, :, :].squeeze().permute(1, 2, 0).unsqueeze(0),
                              align_corners=True))
        elif dim == 3:
            theta = param2theta3(torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                                              device=images[0].device).unsqueeze(0).float(), displacement.size(4),
                                 displacement.size(2), displacement.size(3))
            id = F.affine_grid(theta, displacement[0, 0, :, :, :].squeeze().unsqueeze(0).unsqueeze(0).size())
            warpedIc.append(
                F.grid_sample(images[k], id + displacement[k, :, :, :, :].squeeze().permute(1, 2, 3, 0).unsqueeze(0)))

    return warpedIc


def prolong_displacements(displacement, new_size):
    new_size = new_size[2:]
    dim = len(new_size)
    num_images = displacement.size()[0]

    prolonged_displacement = []
    for k in range(num_images):
        prolonged_tmp = []  ## save single coordinates for later concatenation
        for j in range(dim):
            if dim == 2:
                theta = param2theta(torch.tensor([[1, 0, 0], [0, 1, 0]],
                                                 device=displacement[0, ...].device).unsqueeze(0).float(),
                                    new_size[0], new_size[1])
            elif dim == 3:
                theta = param2theta3(torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                                                  device=displacement[0, ...].device).unsqueeze(0).float(),
                                     new_size[2], new_size[0], new_size[1])
            id = F.affine_grid(theta, ([1, dim] + new_size.tolist()), align_corners=True)
            prolonged_tmp.append(
                F.grid_sample(displacement[k, j, ...].unsqueeze(0).unsqueeze(0), id, align_corners=True))
        prolonged_displacement.append(torch.stack(prolonged_tmp).squeeze())

    return torch.stack(prolonged_displacement)

def warp_images_full_size(images, displacement, full_size, grayscale=True):
    displacement_full = prolong_displacements(displacement, full_size)

    if not grayscale:
        warped_images_full = warp_images_rgb(images, displacement_full)
    else:
        warped_images_full = warp_images(images, displacement_full)

    return warped_images_full


def warp_images_rgb(images, displacement):
    images_r = [img[:, 0, ...].unsqueeze(1) for img in images]
    images_g = [img[:, 1, ...].unsqueeze(1) for img in images]
    images_b = [img[:, 2, ...].unsqueeze(1) for img in images]

    images_r_warped = warp_images(images_r, displacement)
    images_g_warped = warp_images(images_g, displacement)
    images_b_warped = warp_images(images_b, displacement)

    warped_images = []
    for k in range(len(images)):
        tmp_img = torch.zeros_like(images[k])
        tmp_img[:, 0, ...] = images_r_warped[k]
        tmp_img[:, 1, ...] = images_g_warped[k]
        tmp_img[:, 2, ...] = images_b_warped[k]
        warped_images.append(tmp_img)

    return warped_images


def displacement2grid(displacement):
    dim = displacement.size()[1]
    num_images = displacement.size()[0]
    perm_a = np.roll(range(dim + 1), -1).tolist()  # permutation indices for grid dimensions
    perm_b = np.roll(range(dim + 2), -1).tolist()  # perm. indices for final output (purpose: number of images is last)

    grids = []
    for k in range(num_images):
        if dim == 2:
            theta = param2theta(torch.tensor([[1, 0, 0], [0, 1, 0]],
                                             device=displacement[0, ...].device).unsqueeze(0).float(),
                                displacement.size(2), displacement.size(3))
        elif dim == 3:
            theta = param2theta3(torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                                              device=displacement[0, ...].device).unsqueeze(0).float(),
                                 displacement.size(4), displacement.size(2), displacement.size(3))

        id = F.affine_grid(theta, ([1, dim] + torch.tensor(displacement.size()[2:]).tolist()), align_corners=True)
        grid_tmp = id[0, ...] + displacement[k, ...].permute(perm_a).unsqueeze(0).unsqueeze(0)
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


def normalize_displacement(u, dtype=torch.float32, device=torch.device('cpu')):
    # voxel grid to pytorch grid
    m = u.size()
    dim = len(m[2:])

    if dim == 2:
        batches, channels, h, w = u.size()
    elif dim == 3:
        batches, channels, d, h, w = u.size()

    scale = torch.ones(u.size(), dtype=dtype, device=device)
    if dim ==2:
        scale[:, 0, :, :] = scale[:, 0, :, :] / (h - 1) / 2
        scale[:, 1, :, :] = scale[:, 1, :, :] / (w - 1) / 2
    elif dim == 3:
        scale[:, 0, :, :, :] = scale[:, 0, :, :, :] / (d - 1) / 2
        scale[:, 1, :, :, :] = scale[:, 1, :, :, :] / (h - 1) / 2
        scale[:, 2, :, :, :] = scale[:, 2, :, :, :] / (w - 1) / 2

    return (u*scale).to(device)


def denormalize_displacement(u, omega=None, shift=False, dtype=torch.float32, device=torch.device('cpu')):
    # pytorch grid to voxel grid
    m = u.size()
    dim = len(m[2:])

    if dim == 2:
        batches, channels, h, w = u.size()
    elif dim == 3:
        batches, channels, d, h, w = u.size()

    if omega is not None:
        h = omega[3]
        w = omega[1]
        if dim == 3:
            d = omega[5]

    if shift:
        u += 1
        scale_h = h / 2
        scale_w = w / 2
        if dim == 3:
            scale_d = d / 2
    else:
        scale_h = (h - 1) / 2
        scale_w = (w - 1) / 2
        if dim == 3:
            scale_d = (d - 1) / 2

    scale = torch.ones(u.size(), dtype=dtype, device=device)
    if dim ==2:
        scale[:, 0, :, :] = scale[:, 0, :, :] * scale_h
        scale[:, 1, :, :] = scale[:, 1, :, :] * scale_w
    elif dim == 3:
        scale[:, 0, :, :, :] = scale[:, 0, :, :, :] * scale_d
        scale[:, 1, :, :, :] = scale[:, 1, :, :, :] * scale_h
        scale[:, 2, :, :, :] = scale[:, 2, :, :, :] * scale_w

    return (u*scale).to(device)


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
        images[..., k] = image_list[k].squeeze()

    return images


def get_omega(m, h=1, invert=False, centershift=True):
    tmp = np.zeros((m.size, 2))
    tmp[:, 1] = h * m
    y = tmp[0, 1]
    x = tmp[1, 1]
    tmp[0, 1] = x
    tmp[1, 1] = y

    if invert:
        omega = tmp.reshape(2 * m.size)
        omg = omega.copy()
        omega[0] = omg[2]
        omega[1] = omg[3]
        omega[2] = omg[0]
        omega[3] = omg[1]
    else:
        omega = tmp.reshape(2 * m.size)

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
            omega = omega + h / 2

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


def read_dicom(path, dtype=torch.float32, device=torch.device('cpu')):
    ds = dicom.dcmread(path)

    return torch.tensor(ds.pixel_array.astype(np.float), dtype=dtype, device=device)


def save_imagesequence(seq, name='image', path='./img/', type='.tiff', scale=False):
    imgs = []
    n = seq.shape[2]
    fll = len(str(n))
    for k in range(0, seq.shape[2]):
        if type == '.gif':
            imgs.append(seq[:, :, k].astype(np.uint8))
        else:
            if scale:
                sc = (255.0 / seq[:, :, k].max() * (seq[:, :, k] - seq[:, :, k].min())).astype(np.uint8)
            else:
                sc = seq[:, :, k].astype(np.uint8)
            im = Image.fromarray(sc)
            pathlib.Path(path).mkdir(parents=True,
                                          exist_ok=True)  # check if save_path exists. if not: create it and all parental directories
            im.save(path+name+str(k).zfill(fll)+type)

    if type == 'gif':
        imageio.mimsave(path + name + type, imgs)


def save_imagelist(lst, name='image', path='./img/', type='.tiff', scale=False, convert=False):
    k = 0
    n = len(lst)
    fll = len(str(n))
    for x in lst:
        x = x.squeeze().numpy()
        if scale:
            sc = (255.0 / x.max() * (x - x.min())).astype(np.uint8)
        else:
            sc = x.astype(np.uint8)
        im = Image.fromarray(sc)
        if convert:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im.save(path+name+str(k).zfill(fll)+type)
        k += 1


def read_imagelist(path='./', size=None, grayscale=True, type='.png', scale=False, dtype=torch.float32,
                   device=torch.device('cpu')):
    # skipping subdirectories!
    files = sorted([f for f in os.listdir(path) if f.lower().endswith(type) and os.path.isfile(os.path.join(path, f))])

    imagelist = []
    k = 0
    for file in files:
        if file.lower().endswith(type):
            image = torch.tensor(plt.imread(path + file), dtype=dtype, device=device)
            if grayscale:
                if len(image.shape) > 2:
                    image = image[:, :, 0]
            if scale:
                channel_maxima, _ = torch.max(image.reshape(-1, image.shape[-1]), dim=0)
                image = image / channel_maxima
            if not grayscale:
                imagelist.append(image.permute(2, 0, 1).unsqueeze(0))
            else:
                imagelist.append(image.unsqueeze(0).unsqueeze(0))

            if size is not None:
                imagelist[k] = F.interpolate(imagelist[k], size=size[2:].tolist())
            k += 1

    return imagelist


def read_nii(path, dtype=torch.float32, device=torch.device('cpu')):
    data = nib.load(path)
    m = np.array(data.shape)
    dim = data.ndim
    hdr = data.header
    h = np.array(hdr['pixdim'][1:dim+1]) # 0 is left out intentionally!
    omega = get_omega(m, h)

    k = m[-1]
    images = torch.tensor(data.get_fdata(), dtype=dtype, device=device)
    images_list = []
    for k in range(k):
        images_list.append(images[..., k].unsqueeze(0).unsqueeze(0))

    m = np.append([1, 1], m)
    h = torch.ones(1, dim) * torch.tensor(h)

    return images_list, torch.tensor(omega), torch.tensor(m), h


def svd_2x2(arg):
    # counting row first
    d11 = arg[:, 0] * arg[:, 0] + arg[:, 1] * arg[:, 1]
    d12 = arg[:, 0] * arg[:, 2] + arg[:, 1] * arg[:, 3] # = d21
    d22 = arg[:, 2] * arg[:, 2] + arg[:, 3] * arg[:, 3]

    trace = d11 + d22
    det = d11 * d22 - d12 * d12

    num_pixels = arg.size(0)

    eps = 1e-5
    d = torch.sqrt(torch.max(torch.zeros(num_pixels), 0.25 * trace * trace - det) + eps)
    lmax = torch.max(torch.zeros(num_pixels), 0.5 * trace + d)
    lmin = torch.max(torch.zeros(num_pixels), 0.5 * trace - d)
    smax = torch.sqrt(lmax + eps)
    smin = torch.sqrt(lmin + eps)

    return smax, smin


def svd_3x3(arg):
    # compute entries of matrx arg.T * arg (counting is row first)
    d11 = arg[:, 0] * arg[:, 0] + arg[:, 1] * arg[:, 1] + arg[:, 2] * arg[:, 2]
    d22 = arg[:, 3] * arg[:, 3] + arg[:, 4] * arg[:, 4] + arg[:, 5] * arg[:, 5]
    d33 = arg[:, 6] * arg[:, 6] + arg[:, 7] * arg[:, 7] + arg[:, 8] * arg[:, 8]
    d12 = arg[:, 0] * arg[:, 3] + arg[:, 1] * arg[:, 4] + arg[:, 2] * arg[:, 5]  # = d21
    d13 = arg[:, 0] * arg[:, 6] + arg[:, 1] * arg[:, 7] + arg[:, 2] * arg[:, 8]  # = d31
    d23 = arg[:, 3] * arg[:, 6] + arg[:, 4] * arg[:, 7] + arg[:, 5] * arg[:, 8]  # = d32

    # compute coefficients of characteristic polynomial
    a = 1
    b = -(d11 + d22 + d33)
    c = d11 * d22 + d11 * d33 + d22 * d33 - d23 * d23 - d12 * d12 - d13 * d13
    d = -d11 * d22 * d33 + d11 * d23 * d23 + d12 * d12 * d33 - d12 * d23 * d13 - d13 * d12 * d23 + d13 * d13 * d22

    e1, e2, e3 = cardano(a, b, c, d)

    eps = 1e-5
    sings = torch.sqrt(torch.stack((abs(e1), abs(e2), abs(e3))) + eps)

    return sings[0, :], sings[1, :], sings[2, :]


def cardano(A, B, C, D):
    # normalize coefficients
    if A == 1:
        a = B
        b = C
        c = D
    else:  # A != 1:
        a = B / A
        b = C / A
        c = D / A

    p = b - a * a / 3
    q = 2 * a * a * a / 27 - a * b / 3 + c

    eps = 1e-5
    num_pixels = B.size(0)
    discriminant = q * q / 4 + p * p * p / 27

    d = torch.sqrt(torch.max(torch.zeros(num_pixels), discriminant) + eps)
    u = torch.max(torch.zeros(num_pixels), -0.5 * q + d)
    v = torch.max(torch.zeros(num_pixels), -0.5 * q - d)
    u = (u + eps) ** (1 / 3)
    v = (v + eps) ** (1 / 3)

    eps1 = -0.5 + 0.5 * 1j * np.sqrt(3)
    eps2 = -0.5 - 0.5 * 1j * np.sqrt(3)

    # substitutional solutions
    z1 = u + v
    z2 = u * eps1 + v * eps2
    z3 = u * eps2 + v * eps1

    # resubstitution
    x1 = z1 - a / 3
    x2 = z2 - a / 3
    x3 = z3 - a / 3

    return x1, x2, x3


def plot_progress(images, displacement):
    dim = len(images[0].size())-2
    pplt = GReAT4Torch.plot()
    f = plt.figure(1)
    f.clf()
    num_imgs = len(images)

    if dim == 2:
        if num_imgs == 2:
            sbplt = 220
        elif num_imgs > 2:
            sbplt = 230
    elif dim == 3:
        if num_imgs == 2:
            sbplt = 120
        elif num_imgs > 2:
            sbplt = 130

    if dim == 2:
        plt.subplot(sbplt+1)
        plt.imshow(images[0].cpu().squeeze().detach().numpy())
        plt.subplot(sbplt+2)
        plt.imshow(images[1].cpu().squeeze().detach().numpy())
        if num_imgs > 2:
            plt.subplot(sbplt+3)
            plt.imshow(images[2].cpu().squeeze().detach().numpy())
        plt.subplot(sbplt+4) if num_imgs > 2 else plt.subplot(sbplt+3)
        theta = param2theta(torch.tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0).float(),
                                  displacement.size(2), displacement.size(3))
        ident = F.affine_grid(theta, displacement[0, 0, :, :].squeeze().unsqueeze(0).unsqueeze(0).size())
        pplt.plot_grid_2d(
            ident[0, ...].cpu().detach().numpy() + displacement[0, ...].permute(1, 2, 0).cpu().detach().numpy(), )
        plt.subplot(sbplt+5) if num_imgs > 2 else plt.subplot(sbplt+4)
        pplt.plot_grid_2d(
            ident[0, ...].cpu().detach().numpy() + displacement[1, ...].permute(1, 2, 0).cpu().detach().numpy(), )
        if num_imgs > 2:
            plt.subplot(sbplt+6)
            pplt.plot_grid_2d(
                ident[0, ...].cpu().detach().numpy() + displacement[2, ...].permute(1, 2, 0).cpu().detach().numpy(), )
        plt.pause(0.001)
    elif dim == 3:
        n = images[0].size()[-1]/2
        plt.subplot(sbplt+1)
        plt.imshow(images[0].squeeze()[..., int(n)].cpu().squeeze().detach().numpy())
        plt.subplot(sbplt+2)
        plt.imshow(images[1].squeeze()[..., int(n)].cpu().squeeze().detach().numpy())
        if num_imgs > 2:
            plt.subplot(sbplt+3)
            plt.imshow(images[2].squeeze()[..., int(n)].cpu().squeeze().detach().numpy())
        plt.pause(0.001)

def plot_loss(iter, objective, axs, distance=None, regularizer=None):
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    ax1.plot(iter, objective, 'rx')
    ax1.title.set_text('Objective')

    if distance is not None:
        ax2.plot(iter, distance, 'go')
        ax2.title.set_text('Distance')

    if regularizer is not None:
        ax3.plot(iter, regularizer, 'bd')
        ax3.title.set_text('Regularizer')


    plt.pause(0.001)


def landmark_transform(landmarks, displacement, omega, m):
    num_landmarks = landmarks.shape[0]
    dim = landmarks.shape[1]
    m_in = m
    m = m[2:]

    # zero displacement for computation of an initial landmark-error
    if displacement is None:
        m_displacement = np.append(dim, m).tolist()
        displacement = torch.zeros(m_displacement).unsqueeze(0)

    # setup identity grid
    identity = compute_grid(m_in, dtype=displacement[0, ...].dtype,
                            device=displacement[0, ...].device).squeeze().permute(np.roll(range(dim + 1), -1).tolist())

    if dim == 2:
        identity = denormalize_displacement(identity.permute(2, 0, 1).unsqueeze(0),
                                        shift=True, omega=omega).squeeze().permute(1, 2, 0)

        # setup deformation grid
        grid = displacement.squeeze().permute(1, 2, 0) + identity

        # setup interpolation method
        interpolation_x = lambda z: linear_interpolation(z.copy(), grid[..., 0].numpy().copy(), omega.numpy())
        interpolation_y = lambda z: linear_interpolation(z.copy(), grid[..., 1].numpy().copy(), omega.numpy())
        F_inter = lambda z: np.array([interpolation_y(z), interpolation_x(z)]).T.ravel()
    elif dim == 3:
        identity = denormalize_displacement(identity.permute(3, 0, 1, 2).unsqueeze(0),
                                            shift=True, omega=omega).squeeze().permute(1, 2, 3, 0)

        # setup deformation grid
        grid = displacement.squeeze().permute(1, 2, 3, 0) + identity

        # setup interpolation method
        interpolation_x = lambda z: linear_interpolation(z.copy(), grid[..., 0].numpy().copy(), omega.numpy())
        interpolation_y = lambda z: linear_interpolation(z.copy(), grid[..., 1].numpy().copy(), omega.numpy())
        interpolation_z = lambda z: linear_interpolation(z.copy(), grid[..., 2].numpy().copy(), omega.numpy())
        F_inter = lambda z: np.array([interpolation_z(z), interpolation_y(z), interpolation_x(z)]).T.ravel()

    # prepare list of transformed landmarks and flatten grids
    landmarks_transformed = []
    p = identity.numpy().reshape((-1, dim))
    g = grid.numpy().reshape((-1, dim))

    # iterate over all landmarks
    for i in range(0, num_landmarks):
        y = landmarks[i, :]
        min_idx = np.argmin(np.sum((g - y) ** 2, axis=1))  # search for best fit of the landmark on the deformation grid
        x = p[min_idx, :]  # take the same linear index from the identity grid

        # Newton for landmark inversion
        for j in range(0, 100):
            sampled = F_inter(x.copy())
            x = x + (y - sampled)

            if j % 3 == 0:  # only print every third iteration
                print(f'current status of landmark {i}: iter is at {j} of 100, norm: {np.linalg.norm(sampled - y)}')

            if np.linalg.norm(sampled - y) < 1e-8:
                print('Done with landmark ' +str(i)+', '+str(j+1)+' steps for inversion')
                break

        if np.linalg.norm(sampled - y) >= 1e-8:
            print('Fixed-point iteration failed in 100 iterations! Returning initial guess.')
            x = p[min_idx, :]

        landmarks_transformed.append(torch.tensor(x))

    print('Returning transformed landmarks!')
    return torch.stack(landmarks_transformed)


def landmark_accuracy(landmarks_list, h=None):
    # input has to be a list of landmark arrays!
    k = len(landmarks_list)
    m = landmarks_list[0].shape
    y = np.zeros(np.append(m, k))

    for i in range(0, k):
        y[:, :, i] = landmarks_list[i]
    y_bar = np.mean(y, axis=2)
    if h is not None:
        acc = np.sum(np.sqrt(np.sum((h[:, :, None] * y - h[:, :, None] * y_bar[:, :, None]) ** 2,
                                    axis=1))[:, None, :], axis=2)  # / k
    else:
        acc = np.sum(np.sqrt(np.sum((y - y_bar[:, :, None]) ** 2, axis=1))[:, None, :], axis=2)  # / k

    return acc


def linear_interpolation(x, data, omega):
    dim = len(data.shape)
    m = np.array(data.shape)
    h = np.array([(omega[1] - omega[0]) / m[0], (omega[3] - omega[2]) / m[1]])
    if dim == 3:
        h = np.append(h, np.array((omega[5] - omega[4]) / m[2]))

    n = int(len(x) / dim)
    x = x.reshape(n, dim ,order='F')
    Tc = np.zeros(n)

    for k in range(0, dim):
        x[:, k] = (x[:, k] - omega[2 * k]) / h[k] + 0.5

    datashape = np.ones(dim)
    for k in range(0, len(data.shape)):
        datashape[k] = data.shape[k]

    # determine indices of valid points
    Valid = lambda j: ((0 < x[:, j]) & (x[:, j] < m[j] + 1))
    valid = np.empty(n)
    if dim == 1:
        valid = np.where(Valid(0))[0]
    elif dim == 2:
        valid = np.where(Valid(0) & Valid(1))[0]
    elif dim == 3:
        valid = np.where(Valid(0) & Valid(1) & Valid(2))[0]

    # pad data to reduce cases
    pad = 1
    TP = np.zeros(m + 2 * pad)

    P = np.floor(x).astype('int')
    x = x - P
    p = lambda j: P[valid, j]
    xi = lambda j: x[valid, j]

    # increments for linear ordering
    i1 = 1
    i2 = datashape[0].astype('int') + 2 * pad
    i3 = i2 * (datashape[1].astype('int') + 2 * pad)

    # interpolation for different dimensions
    if dim == 1:
        TP[range(0 + pad, m[0] + pad)] = np.reshape(data, (int(m[0]),) ,order='F')
        TP = TP.ravel('F')
        del data
        p = pad + p(0)
        Tc[valid] = TP[p - 1] * (1 - xi(0)) + TP[p] * xi(0)
    if dim == 2:
        TP[(0 + pad):(m[0] + pad), (0 + pad):(m[1] + pad)] = data
        TP = TP.flatten('F')
        del data
        p = ((pad + p(0)) + i2 * (pad + p(1) - 1)).ravel('F')
        part1 = TP[p - 1] * (1 - xi(0)) + TP[p - 1 + i1] * xi(0)
        part2 = TP[p - 1 + i2] * (1 - xi(0)) + TP[p - 1 + i1 + i2] * xi(0)
        Tc[valid] = part1 * (1 - xi(1)) + part2 * xi(1)
    if dim == 3:
        TP[(0 + pad):(m[0] + pad), (0 + pad):(m[1] + pad), (0 + pad):(m[2] + pad)] = data
        TP = TP.reshape(((m[0] + 2 * pad) * (m[1] + 2 * pad) * (m[2] + 2 * pad)), order='F')#ravel('F')
        del data
        p = ((pad + p(0)) + i2 * (pad + p(1) - 1) + i3 * (pad + p(2) - 1)).ravel('F')
        part1 = TP[p - 1] * (1 - xi(0)) + TP[p - 1 + i1] * xi(0)
        part2 = TP[p - 1 + i2] * (1 - xi(0)) + TP[p - 1 + i1 + i2] * xi(0)
        part3 = TP[p - 1 + i3] * (1 - xi(0)) + TP[p - 1 + i1 + i3] * xi(0)
        part4 = TP[p - 1 + i2 + i3] * (1 - xi(0)) + TP[p - 1 + i1 + i2 + i3] * xi(0)
        part12 = part1 * (1 - xi(1)) + part2 * xi(1)
        part34 = part3 * (1 - xi(1)) + part4 * xi(1)
        Tc[valid] = part12 * (1 - xi(2)) + part34 * xi(2)

    return Tc


def save_progress(displacement, params=None, rel_path='saves/', abs_path=None):
    # identify calling script
    filename = inspect.stack()[1].filename
    if 'Windows' in platform.platform():
        path_array = str.split(filename, '\\')
    else:
        path_array = str.split(filename, '/')

    # create absolute path of calling script and caller name
    if abs_path is None:
        abs_path = ''
        for k in range(len(path_array)-1):
            abs_path += path_array[k] + '/'
    caller = path_array[-1][:-3]  # :-3 to remove .py file-extension

    now = datetime.datetime.now()  # get time
    save_path = abs_path + rel_path + caller + '/'  # when copying into driver script: substitute caller with os.path.basename(__file__)[:-3]
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)  # check if save_path exists. if not: create it and all parental directories
    save_name = now.strftime('d' + '%Y%m%d' + '_t' + '%H%M%S') + '_' + caller

    if params is not None:
        json.dump(params, open(save_path + save_name + '.json', 'w'))  # save parameters as .json

    torch.save(displacement, save_path + save_name + '.pt')  # save displacement as .pt

    print('Progress successfully saved as ' + save_path + save_name + ' ...')


def load_progress(file=None, dialog=True):
    if dialog:
        root = tk.Tk()
        root.withdraw()
        file = str.split(filedialog.askopenfilename(), '.')[0]
        path_array = str.split(file, '/')

        abs_path = ''
        for k in range(len(path_array) - 1):
            abs_path += path_array[k] + '/'
        filename = path_array[-1][:]

    print('Loading parameters from ' + file + '.json ...')
    with open(file + '.json') as json_file:
        parameters = json.load(json_file)

    print('Loading displacement from ' + file + '.pt ...')
    displacement = torch.load(file + '.pt')

    print('Successfully loaded data!')

    if dialog:
        return displacement, parameters, abs_path, filename
    else:
        return displacement, parameters


def get_largest_connected_component(labels, k=0): # k indicates the k-largest area, starting from 0
    unique, counts = np.unique(labels, return_counts=True)
    list_seg = list(zip(unique, counts))[1:]  # the 0 label is by default background so take the rest
    largest = max(list_seg, key=lambda x: x[1])[k]
    labels_max = (labels == largest).astype(int)

    return labels_max


def images_to_binary(images_list, lower, upper):
    images_binary = []
    for img in images_list:
        images_binary.append((lower < img) & (img < upper))

    return images_binary


def remove_background(images_list, lower=12, upper=190, hist_eq=False):
    images_binary = images_to_binary(images_list, lower, upper)

    labels = []
    for imgs in images_binary:
        labels.append(torch.tensor(get_largest_connected_component(measurements.label(imgs.numpy())[0])))

    images_segmented = []
    for k in range(len(images_list)):
        images_segmented.append(images_list[k] * labels[k])

    return images_segmented, labels


def remove_background_rgb(rgb_images, lower=12, upper=190, ref_channel=0, hist_eq = False):
    if rgb_images.shape[0] > 1:
        imgs = rgb_images[:, ref_channel, ...].tolist()
    else:
        imgs = [rgb_images[:, ref_channel, ...].squeeze()]

    _, labels = remove_background(imgs, lower, upper, hist_eq)
    rgb_segmented = rgb_images * torch.stack(labels)

    return rgb_segmented


def pad_images_2d(images_list, bound_x=0, bound_y=0, rgb=False):
    # find maximum size in x and y direction
    sizes_x = []
    sizes_y = []
    for k in range(0, len(images_list)):
        sizes_x.append(images_list[k].shape[2])
        sizes_y.append(images_list[k].shape[3])
    max1 = int(max(sizes_x)) + 2 * bound_x
    max2 = int(max(sizes_y)) + 2 * bound_y

    # set up zero image and embed original image in the center
    padded_images_list = []
    k = 0
    for img in images_list:
        if rgb:
            tmp = torch.zeros((1, 3, max1, max2))
        else:
            tmp = torch.zeros((1, 1, max1, max2))

        print('Padding image '+str(k+1))
        bound00 = abs(int((max1-img.shape[2])/2))
        bound01 = img.shape[2]+bound00
        bound10 = abs(int((max2-img.shape[3])/2))
        bound11 = img.shape[3] + bound10
        tmp[:, :, bound00:bound01, bound10:bound11] = img
        padded_images_list.append(tmp)
        k += 1

    return padded_images_list

def remove_padding_2d(images_list, bound_x=0, bound_y=0):
    # find maximum size in x and y direction
    sizes_x_max = []
    sizes_x_min = []
    sizes_y_max = []
    sizes_y_min = []
    for img in images_list:
        idx = img.nonzero()
        sizes_x_max.append(idx[:, 2].max())
        sizes_x_min.append(idx[:, 2].min())
        sizes_y_max.append(idx[:, 3].max())
        sizes_y_min.append(idx[:, 3].min())
    max_x = int(max(sizes_x_max)) + bound_x
    min_x = int(min(sizes_x_min)) - bound_x
    max_y = int(max(sizes_y_max)) + bound_y
    min_y = int(min(sizes_y_min)) - bound_y

    # set up zero image and embed original image in the center
    cropped_images_list = []
    k = 0
    for img in images_list:
        print('Cropping image ' + str(k + 1))
        cropped_images_list.append(img[..., min_x:max_x, min_y:max_y])
        k += 1

    return cropped_images_list
