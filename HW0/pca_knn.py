import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


def parse_args():
	parser = ArgumentParser(description='PCA/KNN')
	parser.add_argument('--dataset-dir', type=str, help='dataset directory', default='./p1_data')
	parser.add_argument('--output-dir', type=str, help='output directory', default='./output')
	return parser.parse_args()


if __name__ == '__main__':
	params = parse_args()
	os.makedirs(params.output_dir, exist_ok=True)

	# Read Data
	X_train, X_test, y_train, y_test = [], [], [], []
	for i in range(1, 41):
		for j in range(1, 11):
			img = mpimg.imread(os.path.join(params.dataset_dir, f'{i}_{j}.png'))
			if j != 10:
				X_train.append(img.flatten())
				y_train.append(i)
			else:
				X_test.append(img.flatten())
				y_test.append(i)
	h, w = img.shape

	# First Question
	fig, axs = plt.subplots(1, 5)
	X_train_mean = np.mean(X_train, axis=0)
	axs[0].set_title('Mean Face', fontsize=8)
	axs[0].imshow(X_train_mean.reshape((h, w)), cmap='gray')
	axs[0].set_axis_off()
	pca = PCA(n_components=None)
	pca.fit(X_train - X_train_mean)
	for i in range(4):
		eig_img = np.dot(pca.singular_values_[i], pca.components_[i]) + X_train_mean
		axs[i+1].set_title(f'Eigenface {i+1}', fontsize=8)
		axs[i+1].imshow(eig_img.reshape((h, w)), cmap='gray')
		axs[i+1].set_axis_off()
	plt.savefig(os.path.join(params.output_dir, '1.png'), bbox_inches='tight', dpi=200)

	# Second and Third Questions
	selected = '2_1.png'
	grids_n = [3, 50, 170, 240, 345]
	fig, axs = plt.subplots(1, 5)
	img = mpimg.imread(os.path.join(params.dataset_dir, selected)).flatten()
	pca_img = pca.transform((img - X_train_mean).reshape(1, -1)).flatten()
	for (i, n) in enumerate(grids_n):
		rec_img = np.dot(pca_img[:n], pca.components_[:n]) + X_train_mean
		mse = mean_squared_error(img, rec_img) * (255 ** 2)
		axs[i].set_title(f'n: {n}\nMSE: {mse:.2f}', fontsize=8)
		axs[i].imshow(rec_img.reshape((h, w)), cmap='gray')
		axs[i].set_axis_off()
	plt.savefig(os.path.join(params.output_dir, '2.png'), bbox_inches='tight', dpi=200)

	# Fourth Question
	grids_k = [1, 3, 5]
	grids_n = [3, 50, 170]
	knns = GridSearchCV(KNeighborsClassifier(), param_grid={'n_neighbors': grids_k}, cv=3)
	results = []
	for n in grids_n:
		pca_X_train = pca.transform(X_train - X_train_mean)
		knns.fit(pca_X_train[:, :n], y_train)
		results.append(knns.cv_results_['mean_test_score'])
	df = pd.DataFrame(results, index=[f'n: {n}' for n in grids_n], columns=[f'k: {k}' for k in grids_k])
	print(df)

	# Fifth Question
	idx = np.unravel_index(np.argmax(results, axis=None), df.shape)
	best_n, best_k = grids_n[idx[0]], grids_k[idx[1]]
	print(f'Best n: {best_n}\nBest k: {best_k}')
	best_knn = KNeighborsClassifier(n_neighbors=best_k)
	pca_X_train = pca.transform(X_train - X_train_mean)
	best_knn.fit(pca_X_train[:, :best_n], y_train)
	pca_X_test = pca.transform(X_test - X_train_mean)
	pred = best_knn.predict(pca_X_test[:, :best_n])
	acc = accuracy_score(y_test, pred)
	print(f'Accuracy: {acc}')
