# Time-Series-data-Classification
This  is a repository for Machine Learning based Time series data classification using Convolutional Neural Networks.
This time series data corresponds to 2D Gaze data consisting of labels for 6 different Gestures( Yes, No, Up, Down, front, Back). After preprocessing and normalizing the data, this preprocessed data is graphically plotted as a type of certain patterns along x and y axis versus time which is further given to Machine learning model.
The model was realized in real time with the help of TOBII EYETRACKER along with its integration in Autonomous driving Real time simulation for navigation using the gaze gestures defined above.

( The repository contains an example of Time series data which is collected in real time with 40 participants imitating these 6 gestures in train.csv and normalized one in norm.csv )
Further, the pattern is represented in the form of image using matplotlib and different processing techniques ( image.csv for Yes gesture)
The training of the model and its testing in model_generation.py
