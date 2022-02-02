import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')
import time
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import GReAT4Torch as grt ## import torch based GReAT as grt

def main():
    start = time.time()
    dtype = torch.float32
    device = torch.device('cpu')

    ######## create circle data using numpy based GReAT-tools ##########################################################
    A = grt.compute_circle(np.array([400, 400]), 100, np.array([200, 200])) / 255
    B = grt.compute_circle(np.array([400, 400]), 100, np.array([250, 220])) / 255
    C = grt.compute_circle(np.array([400, 400]), 100, np.array([170, 200])) / 255

    Ic = [torch.Tensor(A).to(device).unsqueeze(0).unsqueeze(0), torch.Tensor(B).to(device).unsqueeze(0).unsqueeze(0),
          torch.Tensor(C).to(device).unsqueeze(0).unsqueeze(0)]
    m = Ic[0].size() # image size
    num_imgs = len(Ic) # number of images
    dim = 2 # image dimension
    h = torch.ones(1, dim) # pixel spacing
    ####################################################################################################################

    # create registration and transformation objects and levels
    registration = grt.GroupwiseRegistrationMultilevel(min_level=3, max_level=8, dtype=dtype, device=device)
    transformation = grt.NonParametricTransformation(m, num_imgs, dtype=dtype, device=device)
    registration.set_transformation_type(transformation)

    # create distance object and set non-standard parameters
    distance_measure = grt.SqN_pointwise(Ic)
    distance_measure.set_normalization_method('local')
    distance_measure.set_q_parameter(4)
    distance_measure.set_edge_parameter(12)
    # distance_measure.set_distance_weight(1e-2)
    registration.set_distance_measure(distance_measure)

    # create regularizer object and set alpha (regularization weight)
    regularizer = grt.CurvatureRegularizer(h)
    regularizer.set_alpha(26e2)
    registration.set_regularizer(regularizer)

    # set optimizer and non-standard parameters (e.g., maximum iterations)
    optimizer = torch.optim.Adamax(transformation.parameters(), lr=0.2)
    registration.set_optimizer(optimizer)
    registration.set_max_iterations(100)
    registration.set_adaptive_alpha(True, 10)
    registration.set_adaptive_lr(True, 1 / 10)
    registration.set_print_info(True)
    registration.set_plot_progress(False)

    ######## start the registration ####################################################################################
    registration.start()
    ####################################################################################################################

    # get displacement from last level and warp the images
    displacement = transformation.get_displacement()
    warped_images = grt.warp_images(Ic, displacement)

    end = time.time()
    print(f"Registration done in {end-start} seconds.")

    # create plotting instances
    plt_grid = grt.plot()
    plt_original = grt.plot()
    plt_warped = grt.plot()

    # transform images to numpy
    Ic_numpy = grt.image_list2numpy(Ic)
    Ic_warped = grt.image_list2numpy(warped_images)

    # plot the original images, warped images and the grids
    # NOTE: use arrow keys to scroll!
    plt_original.scroll_image_2d(Ic_numpy)
    plt_warped.scroll_image_2d(Ic_warped)
    plt_grid.scroll_grid_2d(transformation.get_grid_numpy())

    # end program when plot windows are closed
    plt.show(block=True)

if __name__ == '__main__':
    main()