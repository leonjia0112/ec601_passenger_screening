
We will be using .aps file for the initial phases of this project as .aps is minimum in the size among all four available formats.
## .aps:-
- Data file order: AYX (angle, vertical axis, horizontal axis)
- Axis Name, Stride, Number of samples, Axis Length
- XAxis, 1, Nx=512, Lx=1.0 meters
- YAxis, 512, Ny=660, Ly=2.0955 meters
- Angular, 337920, Na=16, La=360-degrees

### dataaugmentation.py:
- We used this to create augmented data as the data provided by Kaggle has 10% of the images as threat remaining 90% were threat free.
 
### datadownload.py:
- This file contains python scripts to download the dataset into BU cluster or some other remote server. This is highly customized     format. To use it for your own dataset, good amount of changes will be needed in this file.

### preprocess.py:
- This python file contains the code to preprocess the images and prepare a dataset for the training of images. If run for the full dataset, it will endup generating 121GB data in the PC.

### Sampletrain.py:
- This is the basic training file which contains a basic alexnet and how we can train our machine with different neural network layers. 

### Train_All.py:
 - This file is written to use VGG Net and AlexNet with shuffled data. This will train for the full dataset.

### tsahelper.py:
 - This kernal is originally written by Brian Farrar. It was available in Kaggle kernals as the data best extraction file.
