import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class LinearNet(nn.Module):
    # lr is just for recording the lr this model was trained with
    def __init__(self, in_features, hidden_layers, out_classes, nonlinearity, lr):
        super(LinearNet, self).__init__()
        self.lr = lr
        self.hidden_layer_sizes = hidden_layers
        self.nonlinearity_name = nonlinearity
        self.fc_hidden = nn.ModuleList()
        self.fc1 = nn.Linear(in_features, hidden_layers[0])
        for i in range(len(hidden_layers)-1):
            self.fc_hidden.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.fc2 = nn.Linear(hidden_layers[-1], out_classes)
        if nonlinearity == "sigmoid":
            self.nonlinearity = nn.Sigmoid()
        elif nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()
        
    def forward(self, x):
        x = self.nonlinearity(self.fc1(x))
        for i in range(len(self.fc_hidden)):
            x = self.nonlinearity(self.fc_hidden[i](x))
        x = self.fc2(x)
        return x


def train_model(train_loader, num_features, hidden_layers, nonlinearity, lr, display_training_loss=False):
    model = LinearNet(num_features, hidden_layers, 10, nonlinearity, lr)
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    epochs = 1
    steps_per_epoch = len(train_loader)
    for epoch in range(epochs):
        epoch_loss = np.zeros(steps_per_epoch)
        if display_training_loss:
            print("Epoch " + str(epoch+1) + " started")
        for i, data in enumerate(train_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss[i] = loss.item()
            if display_training_loss:
                if (i % 100 == 99) or (i == 0) or (i+1 == steps_per_epoch): # print every 100 steps
                    print("Epoch " + str(epoch+1) + " Step " + str(i+1) + "/" + str(steps_per_epoch) + " loss = " + str(loss.item()))
        if display_training_loss:
            print("Epoch " + str(epoch+1) + " total loss: " + str(epoch_loss.sum()))
            print("Epoch " + str(epoch+1) + " average loss: " + str(epoch_loss.mean()))
            print()
    return model


def test_model(model, test_loader):
    model.eval()
    num_test = len(test_loader)
    pred_array = torch.zeros(num_test).to(device)
    truth_array = torch.zeros(num_test).to(device)
    for i, sample in enumerate(test_loader):
        img, label = sample
        img = img.to(device)
        truth_array[i] = label
        pred_array[i] = torch.argmax(model(img))
    acc = (pred_array == truth_array).sum() / num_test
    acc = acc.item()
    hidden_layers = model.hidden_layer_sizes
    nonlinearity = model.nonlinearity_name
    lr_str = 'lr {:.0e}'.format(model.lr)
    print('Hidden layer sizes [' + ' '.join(map(str, hidden_layers)) + '], ' + nonlinearity + ' activation, ' + lr_str + ', test accuracy = ' + str(acc))
    return acc


class MnistDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    train_images = np.load('train_images.npy')
    train_labels = np.load('train_labels.npy')
    test_images = np.load('test_images.npy')
    test_labels = np.load('test_labels.npy')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

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


    train_dataset = MnistDataset(X_train_orig, y_train_orig)
    test_dataset = MnistDataset(X_test_orig, y_test_orig)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    num_features = X_train_orig.shape[1]

    lr1 = 1e-3
    # test neural network size
    model = train_model(train_loader, num_features, hidden_layers=[10], nonlinearity="relu", lr=lr1)
    test_model(model, test_loader)
    model = train_model(train_loader, num_features, hidden_layers=[10], nonlinearity="sigmoid", lr=lr1)
    test_model(model, test_loader)

    model = train_model(train_loader, num_features, hidden_layers=[50], nonlinearity="relu", lr=lr1)
    test_model(model, test_loader)
    model = train_model(train_loader, num_features, hidden_layers=[50], nonlinearity="sigmoid", lr=lr1)
    test_model(model, test_loader)

    model = train_model(train_loader, num_features, hidden_layers=[100], nonlinearity="relu", lr=lr1)
    test_model(model, test_loader)
    model = train_model(train_loader, num_features, hidden_layers=[100], nonlinearity="sigmoid", lr=lr1)
    test_model(model, test_loader)

    model = train_model(train_loader, num_features, hidden_layers=[10, 20, 10], nonlinearity="relu", lr=lr1)
    test_model(model, test_loader)
    model = train_model(train_loader, num_features, hidden_layers=[10, 20, 10], nonlinearity="sigmoid", lr=lr1)
    test_model(model, test_loader)

    model = train_model(train_loader, num_features, hidden_layers=[64, 32, 16], nonlinearity="relu", lr=lr1)
    test_model(model, test_loader)
    model = train_model(train_loader, num_features, hidden_layers=[64, 32, 16], nonlinearity="sigmoid", lr=lr1)
    test_model(model, test_loader)

    # test learning rate
    lr2 = 1e-4
    lr3 = 1e-5
    model = train_model(train_loader, num_features, hidden_layers=[100], nonlinearity="relu", lr=lr1)
    test_model(model, test_loader)
    model = train_model(train_loader, num_features, hidden_layers=[100], nonlinearity="sigmoid", lr=lr1)
    test_model(model, test_loader)
    model = train_model(train_loader, num_features, hidden_layers=[100], nonlinearity="relu", lr=lr2)
    test_model(model, test_loader)
    model = train_model(train_loader, num_features, hidden_layers=[100], nonlinearity="sigmoid", lr=lr2)
    test_model(model, test_loader)
    model = train_model(train_loader, num_features, hidden_layers=[100], nonlinearity="relu", lr=lr3)
    test_model(model, test_loader)
    model = train_model(train_loader, num_features, hidden_layers=[100], nonlinearity="sigmoid", lr=lr3)
    test_model(model, test_loader)
