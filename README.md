# GReAT4Torch: [G]roupwise [Re]gistration [A]lgorithms and [T]ools for Torch
see https://github.com/sknR91/GReAT4Torch for details and license issues.

## What is it? 
GReAT4Torch is a python-based (https://www.python.org/) Image Registration toolbox with a focus on registration of multiple images, sometimes referred to as "Groupwise Image Registration".
The toolbox is in many parts heavily inspired by the AIRLAB toolbox (https://github.com/airlab-unibas/airlab) and the concept by the FAIR toolbox (https://github.com/C4IR/FAIR.m).
Please refer to the mentioned git repositories for further information as well as documentation and textbooks.

## Objective
GReAT4Torch intends to create a framework for Groupwise image registration using the python language and especially pyTorch as a basis. This framework is based on the same variational setting as in FAIR but provides slightly different handling for distance measurements. Using such a setting opens up the possibility of registering multiple images instead of just a reference and a template image.

## Requirements
In order to use GReAT4Torch, you need python 3.5 or newer.

The following packages are needed (e.g. using pip for download):

    - torch
    - numpy
    - matplotlib
    - nibabel
    - scipy
    - PyQt5
    - Pillow-PIL
    - pydicom
    
**NOTE**: When used with PyCharm, it is strongly recommended to uncheck 'File > Settings > Tools > Python Scientific > Show plots in tool window' in order to use interactive plotting methods from class "plot"
When interactive plots do not work, please import matplotlib and activate the PyQt-backend.

## Documentation
There is no documentation available yet!

## Author
Kai Brehmer, Institute for Electrical Engineering in Medicine, Universität zu Lübeck, Germany
