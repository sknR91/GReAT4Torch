import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np


class _Transformation(torch.nn.Module):
    def __init__(self, image_size, number_of_images, dtype=torch.float32, device='cpu'):
        super(_Transformation, self).__init__()

        self._dtype = dtype
        self._device = device
        self._dim = len(image_size)-2  ## substract bacthes and channels
        self._num_images = number_of_images
        self._m = np.array(image_size)
        self._reference_grid = None

    def get_displacement_numpy(self):

        if self._dim == 2:
            return torch.unsqueeze(self().detach(), 0).cpu().numpy()
        elif self._dim == 3:
            return self().detach().cpu().numpy()
        elif self._dim == 4:
            print(f"INFO: Method nyi for dim = 4")
            pass

    def get_displacement(self):

        return self().detach()

    def get_current_displacement(self):

        self.get_displacement_numpy()

    def set_reference_grid(self, displacement):

        self._reference_grid = displacement

    def _return_displacement(self, displacement):

        if self._reference_grid is None:
            return displacement
        else:
            return (displacement + self._reference_grid)

class NonParametricTransformation(_Transformation):
    def __init__(self, image_size, number_of_images, dtype=torch.float32, device='cpu'):
        super(NonParametricTransformation, self).__init__(image_size, number_of_images, dtype, device)

        self._tensor_size = [self._num_images, self._dim] + (self._m)[2:].tolist()

        self.transformation_params = Parameter(torch.Tensor(*self._tensor_size))
        self.transformation_params.data.fill_(0)

        self.to(dtype=self._dtype, device=self._device)

        if self._dim == 2:
            self._compute_displacement = self._compute_displacement_2d
        elif self._dim == 3:
            self._compute_displacement = self._compute_displacement_3d
        elif self._dim == 4:
            print(f"INFO: Method nyi for dim = 4")
            pass

    def set_parameters(self, parameters):
        if self._dim == 2:
            self.transformation_params = Parameter(torch.tensor(parameters.transpose(0, 2)))
        elif self._dim == 3:
            self.transformation_params = Parameter(torch.tensor(parameters.transpose(0, 1).transpose(0, 2)
                                                                .transpose(0, 3)))
        elif self._dim == 4:
            print(f"INFO: Method nyi for dim = 4")
            pass

    def _compute_displacement_2d(self):
        return self.transformation_params

    def _compute_displacement_3d(self):
        return self.transformation_params

    def forward(self):
        return self._return_displacement(self._compute_displacement())