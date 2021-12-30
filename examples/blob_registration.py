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

    ######## create blob data using numpy based GReAT-tools ############################################################
    #A = tools.compute_ball(np.array([160, 160, 160]), 20, np.array([80, 80, 80]))
    #B = tools.compute_ball(np.array([160, 160, 160]), 35, np.array([80, 80, 80]))
    #C = tools.compute_ball(np.array([160, 160, 160]), 45, np.array([80, 80, 80]))

    #Ic = [torch.Tensor(A).to(device).unsqueeze(0).unsqueeze(0), torch.Tensor(B).to(device).unsqueeze(0).unsqueeze(0),
    #      torch.Tensor(C).to(device).unsqueeze(0).unsqueeze(0)]
    #m = Ic[0].size()
    #num_imgs = len(Ic)
    #dim = 3
    #h = torch.ones(1, dim)
    ####################################################################################################################

    ######## create circle data using numpy based GReAT-tools ############################################################
    A = grt.compute_circle(np.array([400, 400]), 100, np.array([200, 200]))
    B = grt.compute_circle(np.array([400, 400]), 100, np.array([250, 220]))
    C = grt.compute_circle(np.array([400, 400]), 100, np.array([170, 200]))

    Ic = [torch.Tensor(A).to(device).unsqueeze(0).unsqueeze(0), torch.Tensor(B).to(device).unsqueeze(0).unsqueeze(0),
          torch.Tensor(C).to(device).unsqueeze(0).unsqueeze(0)]
    m = Ic[0].size()
    num_imgs = len(Ic)
    dim = 2
    h = torch.ones(1, dim)
    ####################################################################################################################

    registration = grt.GroupwiseRegistrationMultilevel(min_level=4, max_level=6, dtype=dtype, device=device)
    transformation = grt.NonParametricTransformation(m, num_imgs, dtype=dtype, device=device)
    registration.set_transformation_type(transformation)

    distance_measure = grt.SqN(Ic)
    distance_measure.set_normalization_method('local')
    distance_measure.set_q_parameter(4)
    distance_measure.set_edge_parameter(1e1)
    #distance_measure.set_distance_weight(1e-2)
    registration.set_distance_measure(distance_measure)

    regularizer = grt.CurvatureRegularizer(h)
    regularizer.set_alpha(7e3)
    registration.set_regularizer(regularizer)

    optimizer = torch.optim.Adam(transformation.parameters(), lr=0.2)
    registration.set_optimizer(optimizer)
    registration.set_max_iterations(555)

    ####################################
    registration.start()
    ####################################

    displacement = transformation.get_displacement()
    warped_images = grt.warp_images(Ic, displacement)

    end = time.time()

    print(f"Registration done in {end-start} seconds.")

    Ic_numpy = grt.image_list2numpy(Ic)
    Ic_warped = grt.image_list2numpy(warped_images)
    grt.plot.scrollView3(Ic_numpy)
    grt.plot.scrollView3(Ic_warped)
    grt.plot.scrollGrid2(transformation.get_grid_numpy())
    plt.show(block=True)

if __name__ == '__main__':
    main()