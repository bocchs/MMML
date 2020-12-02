import numpy as np
import matplotlib.pyplot as plt


def pred_value(x, w):
    x = np.reshape(x, (-1,1))
    w = np.reshape(w, (-1,1))
    val = x.T @ w
    val = np.reshape(val, (-1))
    return val

def pred_class(x, w):
    val = pred_value(x, w)
    if val >= 0:
        return 1
    else:
        return -1

def pred_class_one_vs_all(x, w_matrix):
    num_classes = w_matrix.shape[0]
    vals = np.zeros(num_classes)
    for i in range(num_classes):
        w = w_matrix[i]
        vals[i] = pred_value(x, w)
    return np.argmax(vals)

def calculate_loss(X, y, w, lam):
    loss = 0
    for i in range(X.shape[0]):
        hinge = 1 - y[i]*pred_value(X[i], w)
        if hinge < 0:
            hinge = 0
        margin = np.linalg.norm(w, ord=2)**2
        margin = lam * margin
        loss += hinge + margin
    return loss

def calculate_grad(X, y, w, lam):
    grad = np.zeros(len(w))
    for i in range(X.shape[0]):
        if y[i]*pred_value(X[i], w) < 1:
            grad += -y[i]*X[i]
    grad += 2 * lam * w
    return grad



# return train/test data for two classes
def get_binary_data_split(X_train, y_train, X_test, y_test, label1, label2):
    N_train = len(y_train)
    N_test = len(y_test)

    # get class masks
    train_0_mask = y_train == label1*np.ones(N_train)
    train_1_mask = y_train == label2*np.ones(N_train)
    test_0_mask = y_test == label1*np.ones(N_test)
    test_1_mask = y_test == label2*np.ones(N_test)

    # get class train samples
    X_train_0 = X_train[train_0_mask]
    y_train_0 = -np.ones(X_train_0.shape[0])
    X_train_1 = X_train[train_1_mask]
    y_train_1 = np.ones(X_train_1.shape[0])

    # get class test samples
    X_test_0 = X_test[test_0_mask]
    y_test_0 = -np.ones(X_test_0.shape[0])
    X_test_1 = X_test[test_1_mask]
    y_test_1 = np.ones(X_test_1.shape[0])

    # combine different classes into one set
    X_train = np.vstack((X_train_0, X_train_1))
    y_train = np.concatenate((y_train_0, y_train_1))
    X_test = np.vstack((X_test_0, X_test_1))
    y_test = np.concatenate((y_test_0, y_test_1))

    return X_train, y_train, X_test, y_test



def get_one_vs_all_data_split(X_train, y_train, X_test, y_test, label):
    N_train = len(y_train)
    N_test = len(y_test)

    # get class masks
    train_0_mask = y_train != label*np.ones(N_train)
    train_1_mask = y_train == label*np.ones(N_train)
    test_0_mask = y_test != label*np.ones(N_test)
    test_1_mask = y_test == label*np.ones(N_test)

    # get class train samples
    X_train_0 = X_train[train_0_mask]
    y_train_0 = -np.ones(X_train_0.shape[0])
    X_train_1 = X_train[train_1_mask]
    y_train_1 = np.ones(X_train_1.shape[0])

    # get class test samples
    X_test_0 = X_test[test_0_mask]
    y_test_0 = -np.ones(X_test_0.shape[0])
    X_test_1 = X_test[test_1_mask]
    y_test_1 = np.ones(X_test_1.shape[0])

    # combine different classes into one set
    X_train = np.vstack((X_train_0, X_train_1))
    y_train = np.concatenate((y_train_0, y_train_1))
    X_test = np.vstack((X_test_0, X_test_1))
    y_test = np.concatenate((y_test_0, y_test_1))

    return X_train, y_train, X_test, y_test



def train_svm(X_train, y_train, num_steps=1000):
    num_train, m = X_train.shape
    w = np.random.randn(m)
    lr = 1e-4
    batch_size = 128
    lam = 0#1e-5
    print_every = 500
    for i in range(num_steps):
        # sample random batch
        inds = np.random.choice(num_train, batch_size, replace=False)
        X_batch = X_train[inds]
        y_batch = y_train[inds]
        w -= lr * calculate_grad(X_batch, y_batch, w, lam)
        loss = calculate_loss(X_batch, y_batch, w, lam)
        # if i % print_every == 0:
        #     print("Step " + str(i) + " loss = " + str(loss))
    return w



def train_and_test_binary(X_train_orig, y_train_orig, X_test_orig, y_test_orig, class1, class2):
    X_train, y_train, X_test, y_test = get_binary_data_split(X_train_orig, y_train_orig, X_test_orig, y_test_orig, class1, class2)
    w = train_svm(X_train, y_train)
    num_test = len(y_test)
    pred_array = np.zeros(num_test)
    for i in range(num_test):
        pred_array[i] = pred_class(X_test[i], w)
    acc = (pred_array == y_test).sum() / num_test
    print("Test accuracy " + str(class1) + " vs " + str(class2) + ": " + str(acc))



if __name__ == "__main__":
    train_images = np.load('train_images.npy')
    train_labels = np.load('train_labels.npy')
    test_images = np.load('test_images.npy')
    test_labels = np.load('test_labels.npy')

    N_train = len(train_labels)
    N_test = len(test_labels)

    # ----------- flatten 2D images into 1D vectors -----------
    # include bias term at end (+1)
    X_train_orig = np.zeros((N_train, train_images.shape[1] * train_images.shape[2] + 1))
    for i in range(N_train):
        X_train_orig[i] = np.concatenate((train_images[i].flatten(), np.ones(1)))
    y_train_orig = train_labels

    X_test_orig = np.zeros((N_test, test_images.shape[1] * test_images.shape[2] + 1))
    for i in range(N_test):
        X_test_orig[i] = np.concatenate((test_images[i].flatten(), np.ones(1)))
    y_test_orig = test_labels


    # selected binary classification
    train_and_test_binary(X_train_orig, y_train_orig, X_test_orig, y_test_orig, 0, 1)


    # multi-class entire train/test dataset
    w_classifiers = np.zeros((10, X_train_orig.shape[1]))
    for i in range(10):
        X_train, y_train, _, _ = get_one_vs_all_data_split(X_train_orig, y_train_orig, X_test_orig, y_test_orig, i)
        w_classifiers[i] = train_svm(X_train, y_train, num_steps=int(1e4))
    num_test = len(y_test_orig)
    pred_array = np.zeros(num_test)
    for i in range(num_test):
        pred_array[i] = pred_class_one_vs_all(X_test_orig[i], w_classifiers)
    acc = (pred_array == y_test_orig).sum() / num_test
    print("Test accuracy entire dataset: " + str(acc))

