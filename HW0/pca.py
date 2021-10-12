import os
import numpy as np
import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

path = "C:/Users/alex5/碩一(上)/Course/深度學習於電腦視覺/HW/HW0/p1_data"

x_train = []
y_train = []

x_test = []
y_test = []

for filename in os.listdir(path):
	# print(filename)
	# i location
	filename_underscore = filename.find("_")
	# print(filename_underscore)
	# j location
	filename_dot = filename.find(".")

	# i
	image_label = filename[0:filename_underscore]
	# print(image_label)
	# j
	image_number = filename[filename_underscore+1:filename_dot]
	# print(image_number)

	# 讀取每個image的絕對位置
	filepath = os.path.join(path, filename)
	# print(filepath)
	img = plt.imread(filepath, "RGB")
	# 拉成一個row vector
	img = img.reshape(-1)
	
	image_shape = plt.imread(os.path.join(path, "1_1.png")).shape
	# print(image_shape)
	if int(image_number) <= 9:
		x_train.append(img)
		y_train.append(image_label)
	else:
		x_test.append(img)
		y_test.append(image_label)

# Problem 1
# 轉成np array
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

x_mean_face = np.mean(x_train, axis=0)

# image number of training set 
N = x_train.shape[0]

pca = PCA(n_components = N-1)
pca_result = pca.fit(np.subtract(x_train, x_mean_face))

# 獲得前4個eigen值
eigenface1 = pca_result.components_[0].reshape(image_shape)
eigenface2 = pca_result.components_[1].reshape(image_shape)
eigenface3 = pca_result.components_[2].reshape(image_shape)
eigenface4 = pca_result.components_[3].reshape(image_shape)
x_mean_face = np.reshape(x_mean_face, image_shape)

plt.figure(figsize=(15, 10))
first = plt.subplot(1, 5, 1)
plt.imshow(x_mean_face, cmap = "gray")
plt.title("Mean Face", fontsize = 10)

second = plt.subplot(1, 5, 2)
plt.imshow(eigenface1, cmap = "gray")
plt.title("Eigenface 1", fontsize = 10)

third = plt.subplot(1, 5, 3)
plt.imshow(eigenface2, cmap = "gray")
plt.title("Eigenface 2", fontsize = 10)

fourth = plt.subplot(1, 5, 4)
plt.imshow(eigenface3, cmap = "gray")
plt.title("Eigenface 3", fontsize = 10)

fifth = plt.subplot(1, 5, 5)
plt.imshow(eigenface4, cmap = "gray")
plt.title("Eigenface 4", fontsize = 10)

plt.savefig("C:/Users/alex5/碩一(上)/Course/深度學習於電腦視覺/HW/HW0/output/problem1.png")
# plt.show()

# Problem 2 & 3
# my student id: R10942051(odd), use person_2 image_1
original_face = plt.imread(os.path.join(path, str(2) + "_" + str(1) + ".png"), "RGB")
plt.figure(figsize = (25, 15))
original_image = plt.subplot(1, 6, 1)
plt.title("Original Image (person_2 image_1)", fontsize=10)
plt.imshow(original_face, cmap = "gray")

pca_original_image = pca_result.transform(np.subtract(original_face.reshape(1, -1), x_mean_face.reshape(-1)))

j = 2
for i in (3, 50, 170, 240, 345):
	image_pca = np.dot(pca_original_image[0,:i], pca_result.components_[:i]) + x_mean_face.reshape(-1)
	mse = mean_squared_error(image_pca.reshape((1, image_pca.shape[0])), original_face.reshape(1, -1)) * 255 * 255
	image_pca = image_pca.reshape(original_face.shape)
	plt.subplot(1, 6, j)
	plt.title("n = %s, mse = %.6f" % (i, mse))
	plt.imshow(image_pca, cmap = "gray")
	j = j + 1
plt.savefig("C:/Users/alex5/碩一(上)/Course/深度學習於電腦視覺/HW/HW0/output/problem2&3.png")

# plt.show()

# Problem 4
x_train_normalized = pca_result.transform(np.subtract(x_train, x_mean_face.reshape(-1)))
# y_train same as above

parameters = {"n_neighbors": [1, 3, 5]}
KNN = KNeighborsClassifier()
best_result = GridSearchCV(KNN, parameters, cv = 3)

df = dict()

for i in (3, 50, 170):
	best_result.fit(x_train_normalized[:, :i], y_train)
	df["n = " + str(i)] = np.array(best_result.cv_results_["mean_test_score"])

df = pd.DataFrame.from_dict(df, orient = "index")
df.columns = ["k = 1", "k = 3", "k = 5"]
df.index = ["n = 3", "n = 50", "n = 170"]

fig = plt.figure()
ax = fig.add_subplot(111, frame_on = False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

table(ax, df, loc = "center")
plt.savefig("C:/Users/alex5/碩一(上)/Course/深度學習於電腦視覺/HW/HW0/output/problem4.png")

# Problem 5
# my_best_parameters
k = 1
n = 50

pca_test = pca_result.transform(np.subtract(x_test, x_mean_face.reshape(-1)))

# x_train_normalized same as above
# y_train same as above

KNN_optional = KNeighborsClassifier(n_neighbors = k)
KNN_optional.fit(x_train_normalized[:, :n], y_train)
prediction = KNN_optional.predict(pca_test[:, :n])
print("k = ", k)
print("n = ", n)
print("Accuracy on the testing set: ", accuracy_score(y_true = y_test, y_pred = prediction))