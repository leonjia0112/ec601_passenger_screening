# ec601_passenger_screening
passenger screening algorithm for TSA. Aim: Improve the accuracy of existing threat prediction algorithms by TSA.

Dataset for the project is provided by Kggle: https://www.kaggle.com/c/passenger-screening-algorithm-challenge/data

- _.ahi = calibrated object raw data file (2.26GB per file)
- _.aps = projected image angle sequence file (10.3MB per file)
- _.a3d = combined image 3D file (330MB per file)
- _.a3daps = combined image angle sequence file (41.2MB per file)

We will be using .aps file for the initial phases of this project as .aps is minimum in the size among all four available formats.
# .aps:-
- Data file order: AYX (angle, vertical axis, horizontal axis)
- Axis Name, Stride, Number of samples, Axis Length
- XAxis, 1, Nx=512, Lx=1.0 meters
- YAxis, 512, Ny=660, Ly=2.0955 meters
- Angular, 337920, Na=16, La=360-degrees

# datadownload.py:
- This file contains python scripts to download the dataset into BU cluster or some other remote server. This is highly customized     format to use it for your own dataset good amount of changes needed in this file.

# preprocess.py:
- This python file contains the code to preprocess the images and prepare a dataset for the training of images.

# Sampletrain.py:
- This is the basic training file which contains a basic alexnet and how we can train our machine with different neural network layers. 
