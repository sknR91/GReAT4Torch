import torch
import matplotlib.pyplot as plt
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
    A = grt.compute_circle(np.array([400, 400]), 50, np.array([200, 200]))
    B = grt.compute_circle(np.array([400, 400]), 100, np.array([250, 220]))
    C = grt.compute_circle(np.array([400, 400]), 150, np.array([170, 200]))

    Ic = [torch.Tensor(A).to(device).unsqueeze(0).unsqueeze(0), torch.Tensor(B).to(device).unsqueeze(0).unsqueeze(0),
          torch.Tensor(C).to(device).unsqueeze(0).unsqueeze(0)]
    m = Ic[0].size() # image size
    num_imgs = len(Ic) # number of images
    dim = 2 # image dimension
    h = torch.ones(1, dim) # pixel spacing
    ####################################################################################################################

    # create pre alignment
    pre_alignment = grt.GroupwisePrincipalPreAlignment(Ic, dtype=dtype, device=device)
    pre_transformation = grt.PrincipalAffineTransformation(image_size=m, number_of_images=num_imgs, dtype=dtype,
                                                           device=device)
    pre_alignment.set_transformation_type(pre_transformation)

    ######## start the pre-alignment ###################################################################################
    pre_alignment.start()
    displacement_pre_alignment = pre_transformation.get_displacement()
    warped_images_pre_aligned = grt.warp_images(Ic, displacement_pre_alignment)
    ####################################################################################################################

    # create plotting instances
    plt_grid = grt.plot()
    plt_original = grt.plot()
    plt_warped = grt.plot()

    # transform images to numpy
    Ic_numpy = grt.image_list2numpy(Ic)
    Ic_warped_pre_alignment = grt.image_list2numpy(warped_images_pre_aligned)

    # plot the original images, warped images and the grids
    # NOTE: use arrow keys to scroll!
    plt_original.scroll_image_2d(Ic_numpy)
    plt.title('Original Images')
    plt_warped.scroll_image_2d(Ic_warped_pre_alignment)
    plt.title('Pre-Aligned Images')
    plt_grid.scroll_grid_2d(pre_transformation.get_grid_numpy())
    plt.title('Affine Transformation Grids')

    # end program when plot windows are closed
    plt.show(block=False)


    # create registration and transformation objects and levels
    registration = grt.GroupwiseRegistrationMultilevel(min_level=3, max_level=8, dtype=dtype, device=device)
    transformation = grt.NonParametricTransformation(m, num_imgs, dtype=dtype, device=device)
    transformation.set_pre_alignment_displacement(displacement_pre_alignment)
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
    plt_grid_nonparametric = grt.plot()
    plt_warped_nonparametric = grt.plot()

    # transform images to numpy
    Ic_numpy = grt.image_list2numpy(Ic)
    Ic_warped = grt.image_list2numpy(warped_images)

    # plot the original images, warped images and the grids
    # NOTE: use arrow keys to scroll!
    plt_warped_nonparametric.scroll_image_2d(Ic_warped)
    plt.title('Warped Images')
    plt_grid_nonparametric.scroll_grid_2d(transformation.get_grid_numpy())
    plt.title('Non-parametric Transformation Grids')

    # end program when plot windows are closed
    plt.show(block=True)

if __name__ == '__main__':
    main()