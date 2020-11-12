import numpy as np
import skimage.transform # for downsampling image
import matplotlib.pyplot as plt
import time
import argparse

def manhattan_dist(u, v):
	return np.linalg.norm(u-v, ord=1)


def euc_dist(u, v):
	return np.linalg.norm(u-v, ord=2)


def nearest_neigbhors(example, data, K, dist_func):
	# example: length d vector
	# data: the dataset (N x d)
	# K: the number of nearest neighbors to consider
	# dist_func: the type of distance metric to use
	# returns indices of K nearest neighbors in training data and the distances
	min_dist_ra = 1e5*np.ones(K) # index 0 contains smallest dist, index K-1 contains largest dist
	nearest_neigbhor_inds = np.zeros(K, dtype=np.int) # array of K nearest neighbor indices
	for i in range(data.shape[0]):
		dist = dist_func(example, data[i])
		for j in range(K):
			if dist <= min_dist_ra[j]:
				if K > 1:
					for k in range(K-1, j, -1): # shift larger distances toward back of array
						min_dist_ra[k] = min_dist_ra[k-1]
						nearest_neigbhor_inds[k] = nearest_neigbhor_inds[k-1]
				min_dist_ra[j] = dist # insert new distance
				nearest_neigbhor_inds[j] = i
				break
	return nearest_neigbhor_inds, min_dist_ra


def classify_nn(k_nn_inds, y, possible_labels):
	# possible labels are digits 0-9
	# k_nn_inds is indices of nearest neighbors 
	# y is entire label vector
	neighbor_labels = y[k_nn_inds]
	label_out = -1 # most common label
	max_count = 0 # count of most commonly appearing label in neighbors
	for label in possible_labels:
		label_ra = label*np.ones(len(k_nn_inds))
		label_count = (neighbor_labels == label_ra).sum()
		if label_count > max_count:
			label_out = label
			max_count = label_count
	fraction = max_count / len(k_nn_inds) # fraction of neighbors that are the output label
	return label_out, fraction


def test_accuracy(X_train, y_train, X_test, y_test, K, dist_func):
	possible_labels = list(range(10))
	N_train = X_train.shape[0]
	N_test = X_test.shape[0]
	pred = np.zeros(N_test)
	for i in range(N_test):
		print(i)
		example = X_test[i]
		inds, dist = nearest_neigbhors(example, X_train, K, dist_func)
		label_out, frac = classify_nn(inds, y_train, possible_labels)
		pred[i] = label_out
	acc = (pred == y_test).sum() / N_test
	return acc


def pca_proj_matrx(X, num_pc):
	# X: N x d (feature vector lies in row)
	# number of principal components to use
	# return project matrix into lower dimension
	b = np.mean(X, 0)
	X_m = X - b # subtract mean
	U, s, VT = np.linalg.svd(X_m, full_matrices=False)
	V = VT.T
	PC_matrix = V[:,:num_pc]
	return PC_matrix
	

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--PCA', action='store_true', default=False, help='Flag for using PCA')
    parser.add_argument('--num_PC', type=int, default=10, help="Number of principal components to use for PCA")
    parser.add_argument('--img_dim', type=int, default=10, help='Dimension of image to interpolate to')
    parser.add_argument('--K', type=int, default=5, help='Number of nearest neighbors')
    args = parser.parse_args()
    return args




if __name__ == "__main__":
	args = get_args()
	train_images = np.load('train_images.npy')
	train_labels = np.load('train_labels.npy')
	test_images = np.load('test_images.npy')
	test_labels = np.load('test_labels.npy')

	N_train = len(train_labels)
	N_test = len(test_labels)


	# --------- interpolate to smaller size image ------------
	if not args.PCA:
		small_dims = (args.img_dim, args.img_dim) # resize images to this size
		new_dims = (N_train,) + small_dims # concatenate tuples
		new_train_images = np.zeros(new_dims)
		new_test_images = np.zeros(new_dims)

		# interpolate images into smaller dimensions
		for i in range(N_train):
			image = skimage.transform.resize(train_images[i], small_dims)
			image = (255 * image).astype(np.uint8)
			new_train_images[i] = image
		train_images = new_train_images

		for i in range(N_test):
			image = skimage.transform.resize(test_images[i], small_dims)
			image = (255 * image).astype(np.uint8)
			new_test_images[i] = image
		test_images = new_test_images
	


	# ----------- flatten 2D images into 1D vectors -----------
	X_train = np.zeros((N_train, train_images.shape[1] * train_images.shape[2]))
	for i in range(N_train):
		X_train[i] = train_images[i].flatten()
	y_train = train_labels

	X_test = np.zeros((N_test, test_images.shape[1] * test_images.shape[2]))
	for i in range(N_test):
		X_test[i] = test_images[i].flatten()
	y_test = test_labels



	# ---------- PCA -----------
	if args.PCA:
		pca_proj = pca_proj_matrx(X_train, args.num_PC)
		X_train = X_train @ pca_proj
		X_test = X_test @ pca_proj



	N = 100
	X_test = X_test[:N]
	y_test = y_test[:N]


	start_time = time.time()
	test_acc = test_accuracy(X_train, y_train, X_test, y_test, args.K, euc_dist)
	elapsed_time = time.time() - start_time
	print("--- %s seconds ---" % (elapsed_time))
	print(test_acc)
