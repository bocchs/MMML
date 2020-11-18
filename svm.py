import numpy as np

def pred_value(x, w):
    x = np.reshape(x, (-1,1))
    w = np.reshape(w, (-1,1))
    return x.T @ w

def pred_class(x, w):
    val = pred_value(x, w)
    if val >= 0:
        return 1
    else:
        return -1

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


if __name__ == "__main__":
    train_images = np.load('train_images.npy')
    train_labels = np.load('train_labels.npy')
    test_images = np.load('test_images.npy')
    test_labels = np.load('test_labels.npy')

    N_train = len(train_labels)
    N_test = len(test_labels)

    # ----------- flatten 2D images into 1D vectors -----------
    # include bias term at end (+1)
    X_train = np.zeros((N_train, train_images.shape[1] * train_images.shape[2] + 1))
    for i in range(N_train):
        X_train[i] = np.concatenate((train_images[i].flatten(), np.ones(1)))
    y_train = train_labels

    X_test = np.zeros((N_test, test_images.shape[1] * test_images.shape[2] + 1))
    for i in range(N_test):
        X_test[i] = np.concatenate((test_images[i].flatten(), np.ones(1)))
    y_test = test_labels

    # get class masks
    train_0_mask = y_train == 0*np.ones(N_train)
    train_1_mask = y_train == 1*np.ones(N_train)
    test_0_mask = y_test == 0*np.ones(N_test)
    test_1_mask = y_test == 1*np.ones(N_test)

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
    print(X_train_0.shape)
    print(X_train_1.shape)
    print(y_train_0.shape)
    print(y_train_1.shape)
    X_train = np.vstack((X_train_0, X_train_1))
    y_train = np.concatenate((y_train_0, y_train_1))
    X_test = np.vstack((X_test_0, X_test_1))
    y_test = np.concatenate((y_test_0, y_test_1))

    num_train = 1000
    num_test = 2000

    # reduce training and testing size
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]

    X_test = X_test[:num_test]
    y_test = y_test[:num_test]

    m = X_train.shape[1]
    w = np.random.randn(m)

    num_steps = int(1e6)
    lr = 1e-3
    batch_size = 8
    lam=1e-5
    for i in range(num_steps):
        # sample random batch
        inds = np.random.choice(num_train, batch_size, replace=False)
        X_batch = X_train[inds]
        y_batch = y_train[inds]
        w -= lr * calculate_grad(X_batch, y_batch, w, lam)
        loss = calculate_loss(X_batch, y_batch, w, lam)
        print("Step " + str(i) + " loss = " + str(loss))

    pred_array = np.zeros(num_test)
    for i in range(num_test):
        pred_array[i] = pred_class(X_test[i], w)
    acc = (pred_array == y_test).sum() / num_test

    print("Accuracy = " + str(acc))
