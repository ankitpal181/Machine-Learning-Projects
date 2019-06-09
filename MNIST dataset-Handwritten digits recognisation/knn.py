# Import neccessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load MNIST hand written digits dataset using pandas
dataDF = pd.read_csv("../Datasets/MNIST_Hand_Written_Digits.csv")

# Printing dataframe's top 5 rows
dataDF.head()

# Converting data from pandas dataframe to numpy array
data = dataDF.values

# Getting x and y values from the dataframe
x = data[:,1:]
y = data[:,0]

# Splitting data into train and test data
split = int(0.8*x.shape[0])
xTrain = x[:split,:]
yTrain = y[:split]

xTest = x[split:,:]
yTest = y[split:]

# Defining function to plot an image using matplotlib
def drawImg(x):
    img = x.reshape(28,28)
    plt.imshow(img,cmap="gray")
    plt.show()
    return

# Using drawImg function to visiualize an image
drawImg(xTrain[3])
print(yTrain[3])

# Distance function to calculate Euclidian distance between the points
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

# Programming the KNN algorithm
def knn(x,y,queryPoint,k=5):
    # Defining an empty list to store tuples of distances and corresponding label
    vals = []
    m = x.shape[0]
    
    # Looping through x to get all the distances from querypoint
    for i in range(m):
        distance = dist(queryPoint,x[i])
        vals.append((distance,y[i]))
    
    # Sorting the list an retriving top K neighbours
    vals = sorted(vals)
    vals = vals[:k]
    # Getting the count of points from similer class in top K neighbours
    vals = np.array(vals)
    vals_count = np.unique(vals[:,1],return_counts=True)
    # Getting the prediction value for the provided querypoint
    pred = vals_count[0][np.argmax(vals_count[1])]
    return pred

# Predicting a digit in an image using training data
pred = knn(xTrain,yTrain,xTrain[3005])
print(pred)
drawImg(xTrain[3005])