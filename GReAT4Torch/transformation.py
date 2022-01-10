import torch
from torch.nn.parameter import Parameter
import numpy as np
from . import utils


class _Transformation(torch.nn.Module):
    def __init__(self, image_size, number_of_images, dtype=torch.float32, device='cpu'):
        super(_Transformation, self).__init__()

        self._dtype = dtype
        self._device = device
        self._dim = len(image_size)-2  # subtract batches and channels
        self._num_images = number_of_images
        self._reference_grid = None
        self._image_size = image_size
        self._flag_pre_aligned = False

    def get_displacement_numpy(self):

        if self._dim == 2:
            return torch.unsqueeze(self().detach(), 0).cpu().numpy()
        elif self._dim == 3:
            return self().detach().cpu().numpy()
        elif self._dim == 4:
            print(f"INFO: Method nyi for dim = 4")
            pass

    def get_grid_numpy(self):

        return utils.displacement2grid(self()).cpu().detach().numpy()

    def get_displacement(self):

        return self().detach()

    def get_current_displacement(self):

        self.get_displacement_numpy()

    def set_reference_grid(self, displacement):

        self._reference_grid = displacement

    def set_pre_alignment_displacement(self, displacement):

        self._pre_alignment_displacement = displacement
        self._flag_pre_aligned = True

    def set_image_size(self, image_size):

        self._image_size = image_size

    def _return_displacement(self, displacement):

        if self._reference_grid is None:
            return displacement
        else:
            return displacement + self._reference_grid


class NonParametricTransformation(_Transformation):
    def __init__(self, image_size, number_of_images, dtype=torch.float32, device='cpu'):
        super(NonParametricTransformation, self).__init__(image_size, number_of_images, dtype, device)

        self.transformation_params = self._initialize_transformation_parameters(self._image_size)
        self.to(dtype=self._dtype, device=self._device)

    def _initialize_transformation_parameters(self, image_size):
        tensor_size = [self._num_images, self._dim] + np.array(image_size)[2:].tolist()
        transformation_params = Parameter(torch.Tensor(*tensor_size))
        transformation_params.data.fill_(0)
        return transformation_params

    def get_level_displacement(self, image_size):
        if self._flag_pre_aligned:
            displacement = utils.prolong_displacements(self._pre_alignment_displacement, image_size)
            # displacement = Parameter(torch.tensor(displacement))
        else:
            displacement = self._initialize_transformation_parameters(image_size)

        self.set_parameters(displacement)

        return self.transformation_params

    def set_parameters(self, parameters):
        self.transformation_params = Parameter(torch.tensor(parameters))

    def _compute_displacement(self):
        return self.transformation_params

    def forward(self):
        return self._return_displacement(self._compute_displacement())


class PrincipalAffineTransformation(_Transformation):
    def __init__(self, image_size, number_of_images, dtype=torch.float32, device='cpu'):
        super(PrincipalAffineTransformation, self).__init__(image_size, number_of_images, dtype, device)

        self._grid = None
        self._affine_params = None
        self._reference_grid = utils.compute_grid(image_size)
        self._num_images = number_of_images

        self._grid = self._initialize_grid(image_size)
        self.to(dtype=self._dtype, device=self._device)

    def _initialize_grid(self, image_size):

        self._grid = utils.compute_grid(image_size)
        return self._grid

    def set_grid(self, grid):

        self._grid = grid

    def get_grid(self):

        return self._grid

    def set_affine_parameters(self, affine_params):

        self._affine_params = affine_params

    def get_affine_parameters(self):

        return self._affine_params

    def _return_displacement(self, grid):

        displacement = []
        for k in range(self._num_images):
            displacement.append(grid[k].squeeze() - self._reference_grid.squeeze())

        return torch.stack(displacement)

    def forward(self):
        return self._return_displacement(self.get_grid())
