import torchvision.datasets
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def displayImages(images, title="", labels=None, augmented_images=None):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    fig = plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        if labels is not None:
            plt.xlabel(class_names[int(labels[i][0])])
    fig.suptitle("Original", fontsize=16)

    if augmented_images is not None:
        fig2 = plt.figure(2, figsize=(10, 10))
        for i in range(augmented_images.shape[0]):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(augmented_images[i], cmap=plt.cm.binary)
            if labels is not None:
                plt.xlabel(class_names[int(labels[i][0])])
        fig2.suptitle("Augmented", fontsize=16)
    plt.show()


def random_flip(x):
    if np.random.rand() < 0.6:
        x = x[:, ::-1, :]
    return x.copy()


def pad(x, border=4):
    return np.pad(x, [(border, border), (border, border), (0, 0)], mode='reflect')


def pad_and_crop(x, output_size=(32,32)):
    x = pad(x, 4)
    h, w = x.shape[:-1]
    new_h, new_w = output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    x = x[top: top + new_h, left: left + new_w, :]

    return x


def augment(X, K=1):
    augmented_x = X.copy()
    for i in range(X.shape[0]):
        x = X[i, :]
        x = pad_and_crop(x)
        augmented_x[i, :] = random_flip(x)
    return augmented_x


# Test the augment function
dataset = datasets.CIFAR10(root="./datasets", train=True, download=True)
augmented_x = augment(dataset.data[:10])
displayImages(dataset.data[:10], title="Original", augmented_images=augmented_x)


