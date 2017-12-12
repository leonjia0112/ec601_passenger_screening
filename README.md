# ec601_passenger_screening
Passenger Screening algorithm for Transportation Security Adminisistration(TSA) Authority of USA.
### Aim of the project
For every scan in the dataset, we will be predicting the probability that a threat is present in each of 17 body zones. A diagram of the body zone locations is available in the competition files section.

![alt text](https://kaggle2.blob.core.windows.net/competitions/kaggle/6775/media/body_zone_descriptions.png)

### Final versions of codes are in Master branch.

#All other versions of the Codes are available in separate branches. Please Check the below links.
- Web-site demo and development codes are located at:- https://github.com/leonjia0112/ec601_passenger_screening/tree/FrontEndWebDev , https://github.com/leonjia0112/ec601_passenger_screening/tree/Flask_backend
- Images preprocessing codes are located at:- https://github.com/leonjia0112/ec601_passenger_screening/tree/peijia_branch
- Lite version Machine Learning codes are located at:- https://github.com/leonjia0112/ec601_passenger_screening/tree/Vikram_branch. This branch's codes are also located in "Training and preprocessing" folder in master branch.


Dataset for the project is provided by Kggle: https://www.kaggle.com/c/passenger-screening-algorithm-challenge/data

- _.ahi = calibrated object raw data file (2.26GB per file)
- _.aps = projected image angle sequence file (10.3MB per file)
- _.a3d = combined image 3D file (330MB per file)
- _.a3daps = combined image angle sequence file (41.2MB per file)

We will be using .aps file for the initial phases of this project as .aps is minimum in the size among all four available formats.
## .aps:-
- Data file order: AYX (angle, vertical axis, horizontal axis)
- Axis Name, Stride, Number of samples, Axis Length
- XAxis, 1, Nx=512, Lx=1.0 meters
- YAxis, 512, Ny=660, Ly=2.0955 meters
- Angular, 337920, Na=16, La=360-degrees

## Training:-
 - We trained the data for AlexNet and VGGNet. Below are the accuracies for each threat zone.
 
| Threat Zone | Accuracy % |
| --- | --- |
| Zone1	| 91.99 |
| Zone2	| 92.95 |
| Zone3	| 94.11 |
| Zone4	| 93.14 |
| Zone5	| 93.04 |
| Zone6	| 93.17 |
| Zone7	| 93.54 |
| Zone8	| 94.11 |
| Zone9	| 93.84 |
| Zone10	| 92.73 |
| Zone11	| 95.18 |
| Zone12	| 93.55 |
| Zone13	| 93.68 |
| Zone14	| 95.14 |
| Zone15	| 93.66 |
| Zone16	| 93.63 |
| Zone17	| 93.37 |

## Unit Test:-
### We have finished unit test in our website,here is our unit test sheet:
 -https://docs.google.com/spreadsheets/d/1TRVQTAOrh1yoBVjL5CXdc05WejdTG9oMAG8MyT0Buc0/edit?usp=sharing
