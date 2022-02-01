import torch
import numpy as np
import torch.nn.functional as F
from . import utils
import GReAT4Torch
import matplotlib.pyplot as plt


class _Registration():
    def __init__(self, dtype=torch.float32, device='cpu'):
        self._dtype = dtype
        self._device = device

        # set distance measure
        self._distance_measure = None

        # set regularizer
        self._regularizer = None

        # set optimizer
        self._optimizer = None
        self._max_iterations = 250

        # set displacement
        self._displacement = None

        # set transformation type
        self._transformation_type = None

        # set flag for printing iteration information and plotting the progress
        self._print_info = True
        self._plot_progress = False

    def set_distance_measure(self, distance):
        self._distance_measure = distance

    def set_regularizer(self, regularizer):
        self._regularizer = regularizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def set_max_iterations(self, max_iterations):
        self._max_iterations = max_iterations

    def set_transformation_type(self, transformation_type):
        self._transformation_type = transformation_type

    def set_print_info(self, flag):
        self._print_info = flag

    def set_plot_progress(self, flag):
        self._plot_progress = flag


class _GroupwiseRegistration(_Registration):
    def __init__(self, dtype=torch.float32, device='cpu'):
        super(_GroupwiseRegistration, self).__init__(dtype, device)

        # set images (previously defined in numpy-based GReAT as "Ic")
        self._images = None

    def set_images(self, images):
        self._images = images


class _GroupwisePrincipalPreAlignment(_Registration):
    def __init__(self, images, dtype=torch.float32, device='cpu'):
        super(_GroupwisePrincipalPreAlignment, self).__init__(dtype, device)

        # set images (previously defined in numpy-based GReAT as "Ic")
        self._images = images

    def set_images(self, images):
        self._images = images


class GroupwiseRegistration(_GroupwiseRegistration):
    def __init__(self, dtype=torch.float32, device='cpu'):
        super(GroupwiseRegistration, self).__init__(dtype, device)

    def _driver(self):
        self._optimizer.zero_grad()
        displacement = self._transformation_type()

        if self._plot_progress:
            utils.plot_progress(utils.warp_images(self._images, displacement), displacement)

        # compute distance between images
        dist = self._distance_measure(displacement)

        # compute the regularizer
        nI = self._transformation_type._num_images
        regul = 0
        for k in range(nI):
            regul += self._regularizer(displacement[k, ...].unsqueeze(0))

        # compute value of objective function as sum of distance and regularizer
        obj = dist + regul

        # some output
        if self._print_info:
            print(f"distance + regularizer = objective: {dist} + {regul} = {obj}")

        obj.backward()

        return obj, dist, regul

    def start(self):
        print("="*30, " Starting groupwise image registration ", "="*30)
        print("-- Distance Measure: ", self._distance_measure.name)
        print("-- Regularizer: ", self._regularizer.name)

        for iter in range(self._max_iterations):
            if self._print_info:
                print(f" Iter {iter}: ", end='', flush=True)
            obj, dist, regul = self._optimizer.step(self._driver)

            if self._plot_progress >= 2:
                if iter == 0:
                    # ax = plt.axes()
                    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
                utils.plot_loss(iter, obj.detach(), axs=[ax1, ax2, ax3],
                                distance=dist.detach(), regularizer=regul.detach())
                ax1 = ax1; ax2 = ax2; ax3 = ax3


class GroupwiseRegistrationMultilevel(_GroupwiseRegistration):
    def __init__(self, min_level, max_level, dtype=torch.float32, device='cpu'):
        super(GroupwiseRegistrationMultilevel, self).__init__(dtype, device)

        self._min_level = min_level
        self._max_level = max_level
        self._max_level_data = None
        self._adaptive_alpha = False
        self._adaptive_lr = False
        self._adaptive_iter = False

    def set_adaptive_iter(self, iter_list):
        self._max_iterations = iter_list
        self._adaptive_iter = True

    def set_adaptive_alpha(self, flag, rate=50):
        self._adaptive_alpha = flag
        self._adaptive_alpha_rate = rate

    def set_adaptive_lr(self, flag, rate=50):
        self._adaptive_lr = flag
        self._adaptive_lr_rate = rate

    def _driver(self):
        self._optimizer.zero_grad()
        displacement = self._transformation_type()

        if self._plot_progress >= 1 and self._plot_progress <= 2:
            utils.plot_progress(utils.warp_images(self._images, displacement), displacement)

        # compute distance between images
        dist = self._distance_measure(displacement)

        # compute the regularizer
        nI = self._transformation_type._num_images
        regul = 0
        for k in range(nI):
            regul += self._regularizer(displacement[k, ...].unsqueeze(0))

        # compute value of objective function as sum of distance and regularizer
        obj = dist + regul

        # some output
        if self._print_info:
            print(f"distance + regularizer = objective: {dist} + {regul} = {obj}")

        obj.backward(retain_graph=True)

        return obj, dist, regul

    def _get_max_level_parameters(self, max_level, image_size):
        image_size = torch.tensor(image_size[2:])
        max_level_data = torch.ceil(torch.log2(torch.min(image_size))).int()
        if max_level > max_level_data:
            print(f"Warning: chosen max_level {max_level} is too large, chose {max_level_data} instead")
        max_level = np.min((max_level, max_level_data)).astype('int')
        image_size_level = torch.round(image_size / 2 ** (np.max((1, abs(max_level_data - max_level))))).int()
        num_pixels_level = torch.prod(image_size_level)

        return max_level, image_size_level, num_pixels_level, max_level_data

    def _compute_multilevel_data(self):
        images = self._images
        image_size = torch.tensor(self._transformation_type._image_size)
        num_images = self._transformation_type._num_images
        min_level = self._min_level
        m_level = image_size
        dim = image_size.numel()-2

        if dim == 2:
            interpolation_mode = 'bicubic'
        elif dim == 3:
            interpolation_mode = 'trilinear'

        max_level, image_size_level, num_pixels_level, max_level_data = \
            self._get_max_level_parameters(self._max_level, image_size)
        self._max_level = max_level
        self._max_level_data = max_level_data

        data_ML = [None] * (max_level+1)
        m_ML = [None] * (max_level+1)
        images_res = images
        m_res = image_size
        for level in range(max_level_data, min_level-1, -1):
            if level == max_level_data: ## take original data to resize to max_level
                Ic_level = []
                for k in range(num_images):
                    tmp = F.interpolate(images[k], size=image_size[2:].tolist(), mode=interpolation_mode)
                    Ic_level.append(tmp)
                images_res = Ic_level
                m_res = m_level
            else: ## starting from the defined max_level to resize all smaller levels down to min_level
                m_resm1 = torch.round(m_res/2).int()
                Ic_level = []
                for k in range(0, num_images):
                    tmp = F.interpolate(images_res[k], size=m_resm1[2:].tolist(), mode=interpolation_mode)
                    Ic_level.append(tmp)
                m_res = m_resm1
                images_res = Ic_level
            if level <= max_level:
                data_ML[level] = Ic_level
                m_ML[level]  = m_res
            m_level = m_res
        return data_ML, m_ML

    def start(self):
        self.set_images(self._distance_measure._images)
        images_levels, image_sizes_levels = self._compute_multilevel_data()
        alpha = self._regularizer.get_alpha()

        print("=" * 30, " Starting multilevel groupwise image registration ", "=" * 30)
        print("-- Distance Measure: ", self._distance_measure.name)
        print("-- Regularizer: ", self._regularizer.name)
        print(f"-- Level(s): {self._min_level} to {self._max_level}")
        print(f"-- Level of full resolution: {self._max_level_data}\n")

        # iterate over all levels
        for level in range(self._min_level, self._max_level+1):
            print(f"\n", '='*30, f" Level {level}", '='*30)
            self._distance_measure.set_images(images_levels[level])
            #self._transformation_type.set_image_size(image_sizes_levels[level])

            if level == self._min_level:
                # displacement = self._transformation_type._initialize_transformation_parameters(image_sizes_levels[level])
                displacement = self._transformation_type.get_level_displacement(image_sizes_levels[level])
                self._transformation_type.set_parameters(displacement)

            if level != self._min_level:
                # do prolongation of grids
                displacement = utils.prolong_displacements(displacement, image_sizes_levels[level])
                self._transformation_type.set_parameters(displacement)

            # fetch all attributes from the original optimizer to pass it to a freshly initialized instance
            optim_attributes = { _key : self._optimizer.param_groups[0][_key] for _key in self._optimizer.param_groups[0] }
            del optim_attributes['params']

            # reduce learning rate or alpha for each higher level
            if level != self._min_level:
                if self._adaptive_lr:
                    optim_attributes['lr'] *= self._adaptive_lr_rate
                if self._adaptive_alpha:
                    alpha *= self._adaptive_alpha_rate
                    self._regularizer.set_alpha(alpha)

            # reinitialize the optimizer instance with new "params" and all pre-set attributes!
            self._optimizer.__init__(self._transformation_type.parameters(), **optim_attributes)

            if self._adaptive_iter:
                max_iter = self._max_iterations[level]
            else:
                max_iter = self._max_iterations

            for iter in range(max_iter):
                if self._print_info:
                    print(f" Iter {iter}: ", end='', flush=True)
                obj, dist, regul = self._optimizer.step(self._driver)

                if False: #self._adaptive_lr:
                    if iter <= 50:
                        self._optimizer.param_groups[0]['lr'] = self._optimizer.param_groups[0]['lr'] * 1.01
                    elif iter > 50 and iter <= 100:
                        self._optimizer.param_groups[0]['lr'] = self._optimizer.param_groups[0]['lr'] * 1.02

                if self._plot_progress >= 2:
                    if iter == 0:
                        #ax = plt.axes()
                        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
                    utils.plot_loss(iter, obj.detach(), axs=[ax1, ax2, ax3],
                                    distance=dist.detach(), regularizer=regul.detach())
                    ax1 = ax1; ax2 = ax2; ax3 = ax3

            displacement = self._transformation_type.get_displacement()


class GroupwisePrincipalPreAlignment(_GroupwisePrincipalPreAlignment):
    def __init__(self, images, dtype=torch.float32, device='cpu'):
        super(GroupwisePrincipalPreAlignment, self).__init__(images, dtype, device)

        self.set_images(images)

    def _driver(self):
        grid = self._transformation_type.get_grid()
        num_images = self._transformation_type._num_images
        image_size = self._transformation_type._image_size
        m = torch.tensor(image_size[2:])
        dim = len(image_size)-2
        perm = np.roll(range(dim + 1), -1).tolist()

        center_size = torch.zeros(dim)  # the center is simply zero because of the normalized grids; otherwise: m / 2
        C = []; Cov = []; S = []; U = []

        covM = torch.zeros((dim, dim))
        flat_grid = grid.squeeze().permute(perm).view(-1, dim)  # get rid of unnecessary dimensions and flatten grid
        # iterate over all images
        for k in range(num_images):
            flat_image = self._images[k].squeeze().contiguous().view(-1)
            C.append(torch.sum(flat_grid * flat_image[:, None], dim=0) / torch.sum(flat_image))  # comp. center of mass
            Cov.append(utils.compute_covariance_matrix(flat_grid - C[k], flat_image))
            tmp_s, tmp_u = torch.eig(Cov[k], eigenvectors=True)
            S.append(torch.sqrt(tmp_s[:, 0].squeeze()))
            U.append(tmp_u.squeeze())
            covM += torch.diag(S[k] ** 2)

        covM /= num_images
        S_M, U_M = torch.eig(covM, eigenvectors=True)
        S_M = torch.sqrt(S_M[:, 0].squeeze())

        y = []; w = []

        new_size = m.tolist()
        new_size.insert(0, dim)

        for k in range(num_images):
            A = U[k] @ torch.diag(S[k]) @ torch.diag(1/S_M) @ U_M.T
            b = C[k] - A @ center_size.T
            w.append(torch.cat((A.view(-1), b)))
            y.append((A @ flat_grid.T + b[:, None]).view(new_size).unsqueeze(0))

            # some output
            if self._print_info:
                print(f"Pre-aligning image {k+1} of {num_images}.")

        if self._print_info:
            print(f"Done with pre-alignment of {num_images} images!")

        return y, w

    def start(self):
        print("="*30, " Starting groupwise pre-alignment based on principal components ", "="*30)

        y, w = self._driver()
        self._transformation_type.set_affine_parameters(w)
        self._transformation_type.set_grid(y)
