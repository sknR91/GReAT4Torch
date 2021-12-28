import torch
import matplotlib.pyplot as plt
import time
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import GReAT4Torch as grt ## import torch based GReAT as grt

sys.path.append('/Users/kai/Documents/great')
import GReAT ## import numpy based GReAT for some helpful tools

def main():
    start = time.time()
    dtype = torch.float32
    device = torch.device('cpu')

    tools = GReAT.Tools()
    pplt = GReAT.plot()

    ######## create blob data using numpy based GReAT-tools ############################################################
    #A = tools.compute_ball(np.array([160, 160, 160]), 20, np.array([80, 80, 80]))
    #B = tools.compute_ball(np.array([160, 160, 160]), 35, np.array([80, 80, 80]))
    #C = tools.compute_ball(np.array([160, 160, 160]), 45, np.array([80, 80, 80]))

    #Ic = [torch.Tensor(A).to(device).unsqueeze(0).unsqueeze(0), torch.Tensor(B).to(device).unsqueeze(0).unsqueeze(0),
    #      torch.Tensor(C).to(device).unsqueeze(0).unsqueeze(0)]
    #Ic = [torch.Tensor(A).to(device), torch.Tensor(B).to(device), torch.Tensor(C).to(device)]
    #m = Ic[0].size()
    #dim = 3
    #h = torch.ones(1, dim)
    ####################################################################################################################

    ######## create circle data using numpy based GReAT-tools ############################################################
    A = tools.compute_circle(np.array([20, 20]), 5, np.array([10, 10]))
    B = tools.compute_circle(np.array([20, 20]), 6, np.array([12, 12]))
    C = tools.compute_circle(np.array([20, 20]), 7, np.array([10, 10]))

    Ic = [torch.Tensor(A).to(device).unsqueeze(0).unsqueeze(0), torch.Tensor(B).to(device).unsqueeze(0).unsqueeze(0),
          torch.Tensor(C).to(device).unsqueeze(0).unsqueeze(0)]
    # Ic = [torch.Tensor(A).to(device), torch.Tensor(B).to(device), torch.Tensor(C).to(device)]
    m = Ic[0].size()
    num_imgs = len(Ic)
    dim = 2
    h = torch.ones(1, dim)
    ####################################################################################################################

    registration = grt.GroupwiseRegistration(dtype=dtype, device=device)
    transformation = grt.NonParametricTransformation(m, num_imgs, dtype=dtype, device=device)
    registration.set_transformation_type(transformation)

    distance_measure = grt.SqN(Ic)
    distance_measure.set_normalization_method('local')
    #distance_measure.set_distance_weight(1e-2)
    registration.set_distance_measure(distance_measure)

    regularizer = grt.CurvatureRegularizer(h)
    regularizer.set_alpha(1e2)
    registration.set_regularizer(regularizer)

    optimizer = torch.optim.Adamax(transformation.parameters(), lr=0.01)
    registration.set_optimizer(optimizer)
    registration.set_max_iterations(1000)

    ####################################
    registration.start()
    ####################################

    displacement = transformation.get_displacement()
    warped_images = grt.warp_images(Ic, displacement)

    end = time.time()

    print(f"Registration done in {end-start} seconds.")
    #pplt.scrollView3(Ic.cpu()[0, 0, ...].detach().numpy())
    #pplt.scrollView3(warped_images.cpu()[0,0,...].detach().numpy())
    plt.show(block=True)

if __name__ == '__main__':
    main()