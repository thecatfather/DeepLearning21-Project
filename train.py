from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

X = datasets.CIFAR10(root='./datasets', train=True, download=True)


def displayImages(images, labels=None):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        if labels is not None:
            plt.xlabel(class_names[int(labels[i][0])])
    plt.show()


displayImages(X.data[:10])
