import struct
import numpy as np
import matplotlib.pyplot as plt

# save images as N x rows x cols
def read_save_train_images():
	train_images_file = 'data_files/train-images.idx3-ubyte'
	with open(train_images_file, 'rb') as f:
		magic = f.read(4)
		magic = struct.unpack('>i', magic)[0]
		num_examples = f.read(4)
		num_examples = struct.unpack('>i', num_examples)[0]
		num_rows = f.read(4)
		num_rows = struct.unpack('>i', num_rows)[0]
		num_cols = f.read(4)
		num_cols = struct.unpack('>i', num_cols)[0]

		# fill images array
		images = np.zeros((num_examples, num_rows, num_cols), dtype=np.uint8)
		for i in range(num_examples):
			for r in range(num_rows):
				row = f.read(num_cols)
				row = struct.unpack(num_cols*'B', row)
				images[i, r] = row

		np.save('train_images.npy', images)



def read_save_train_labels():
	train_labels_file = 'data_files/train-labels.idx1-ubyte'
	with open(train_labels_file, 'rb') as f:
		magic = f.read(4)
		magic = struct.unpack('>i', magic)[0]
		num_examples = f.read(4)
		num_examples = struct.unpack('>i', num_examples)[0]


		# fill labels array
		labels = np.zeros((num_examples), dtype=np.uint8)
		for i in range(num_examples):
			the_label = f.read(1)
			the_label = struct.unpack('B', the_label)[0]
			labels[i] = the_label

		np.save('train_labels.npy', labels)


# save images as N x rows x cols
def read_save_test_images():
	test_images_file = 'data_files/t10k-images.idx3-ubyte'
	with open(test_images_file, 'rb') as f:
		magic = f.read(4)
		magic = struct.unpack('>i', magic)[0]
		num_examples = f.read(4)
		num_examples = struct.unpack('>i', num_examples)[0]
		num_rows = f.read(4)
		num_rows = struct.unpack('>i', num_rows)[0]
		num_cols = f.read(4)
		num_cols = struct.unpack('>i', num_cols)[0]

		# fill images array
		images = np.zeros((num_examples, num_rows, num_cols), dtype=np.uint8)
		for i in range(num_examples):
			for r in range(num_rows):
				row = f.read(num_cols)
				row = struct.unpack(num_cols*'B', row)
				images[i, r] = row

		np.save('test_images.npy', images)



def read_save_test_labels():
	test_labels_file = 'data_files/t10k-labels.idx1-ubyte'
	with open(test_labels_file, 'rb') as f:
		magic = f.read(4)
		magic = struct.unpack('>i', magic)[0]
		num_examples = f.read(4)
		num_examples = struct.unpack('>i', num_examples)[0]


		# fill labels array
		labels = np.zeros((num_examples), dtype=np.uint8)
		for i in range(num_examples):
			the_label = f.read(1)
			the_label = struct.unpack('B', the_label)[0]
			labels[i] = the_label

		np.save('test_labels.npy', labels)




if __name__ == "__main__":
	read_save_train_images()
	read_save_train_labels()
	read_save_test_images()
	read_save_test_labels()

	# train_images = np.load('train_images.npy')
	# train_labels = np.load('train_labels.npy')
	# test_images = np.load('test_images.npy')
	# test_labels = np.load('test_labels.npy')


