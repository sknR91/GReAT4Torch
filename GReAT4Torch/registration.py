import torch

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