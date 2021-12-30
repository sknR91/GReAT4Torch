import torch
import numpy as np
import torch.nn.functional as F
from . import utils

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

class _GroupwiseRegistration(_Registration):
    def __init__(self, dtype=torch.float32, device='cpu'):
        super(_GroupwiseRegistration, self).__init__(dtype, device)

        # set images (previously defined in numpy-based GReAT as "Ic")
        self._images = None

    def set_images(self, images):
        self._images = images

class GroupwiseRegistration(_GroupwiseRegistration):
    def __init__(self, dtype=torch.float32, device='cpu'):
        super(GroupwiseRegistration, self).__init__(dtype, device)

    def _driver(self):
        self._optimizer.zero_grad()
        displacement = self._transformation_type()

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
        print(f"Current value of objective function D + R = J: {dist} + {regul} = {obj}")

        obj.backward()

        return obj

    def start(self):
        for iter in range(self._max_iterations):
            print(f"Iteration {iter}: ", end='', flush=True)
            obj = self._optimizer.step(self._driver)

class GroupwiseRegistrationMultilevel(_GroupwiseRegistration):
    def __init__(self, min_level, max_level, dtype=torch.float32, device='cpu'):
        super(GroupwiseRegistrationMultilevel, self).__init__(dtype, device)

        self._min_level = min_level
        self._max_level = max_level

    def _driver(self):
        self._optimizer.zero_grad()
        displacement = self._transformation_type()

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
        print(f"Current value of objective function D + R = J: {dist} + {regul} = {obj}")

        obj.backward()

        return obj

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

        max_level, image_size_level, num_pixels_level, max_level_data = \
            self._get_max_level_parameters(self._max_level, image_size)
        self._max_level = max_level

        data_ML = [None] * (max_level+1)
        m_ML = [None] * (max_level+1)
        images_res = images
        m_res = image_size
        for level in range(max_level_data, min_level-1, -1):
            if level == max_level_data: ## take original data to resize to max_level
                Ic_level = []
                for k in range(num_images):
                    tmp = F.interpolate(images[k], size=image_size[2:].tolist(), mode='bicubic')
                    Ic_level.append(tmp)
                images_res = Ic_level
                m_res = m_level
            else: ## starting from the defined max_level to resize all smaller levels down to min_level
                m_resm1 = torch.round(m_res/2).int()
                Ic_level = []
                for k in range(0, num_images):
                    tmp = F.interpolate(images_res[k], size=m_resm1[2:].tolist(), mode='bicubic')
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

        # iterate over all levels
        for level in range(self._min_level, self._max_level+1):
            print(f"Level {level}, ", end='', flush=True)
            self._distance_measure.set_images(images_levels[level])
            #self._transformation_type.set_image_size(image_sizes_levels[level])

            if level == self._min_level:
                displacement = self._transformation_type._initialize_transformation_parameters(image_sizes_levels[level])
                self._transformation_type.set_parameters(displacement)

            if level != self._min_level:
                # do prolongation of grids
                displacement = utils.prolong_displacements(displacement, image_sizes_levels[level])
                self._transformation_type.set_parameters(displacement)

            # fetch all attributes from the original optimizer to pass it to a freshly initialized instance
            optim_attributes = { _key : self._optimizer.param_groups[0][_key] for _key in self._optimizer.param_groups[0] }
            del optim_attributes['params']
            if level != self._min_level:
                optim_attributes['lr'] = optim_attributes['lr'] / 10

                # reinitialize the optimizer instance with new "params" and all pre-set attributes!
            self._optimizer.__init__(self._transformation_type.parameters(), **optim_attributes)

            for iter in range(self._max_iterations):
                print(f" Iteration {iter}: ", end='', flush=True)
                obj = self._optimizer.step(self._driver)

            displacement = self._transformation_type.get_displacement()