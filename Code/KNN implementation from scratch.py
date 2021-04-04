import pandas as pd
import numpy as np
df_x= pd.read_excel('ex3d1.xlsx', 'X', header=None)
df_x.head()


#image representatio of the data
import matplotlib.pyplot as plt
plt.imshow(np.array(df.iloc[1, :]).reshape(20,20))

df_y= pd.read_excel('ex3d1.xlsx', 'y', header=None)
#df_y.size
train_label=df_y[0]
#train_label.head()
#train_label.size
#df_y.size
#df_y.head()

x_train = np.array(train_label)
x_train[1705]

df_x.shape

#df_new_x=pd.concat([df_x,train_label],axis=1)
df_new_x=df_x


#added label column to teh dataset as 400th column
df_new_x[400]=x_train

df_new_x.head()
#df_new_x.shape

dataset = np.array(df_new_x)
dataset.shape

# Example of making predictions
from math import sqrt

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# Test distance function

prediction = predict_classification(dataset, dataset[1705], 3)
print('Expected %d, Got %d.' % (dataset[1705][-1], prediction))
