import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np


def displayImages(images, title1="Original", title2="Augmented", labels=None, augmented_images=None):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    fig = plt.figure(figsize=(images.shape[0], images.shape[0]))
    for i in range(images.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        if labels is not None:
            plt.xlabel(class_names[int(labels[i])])
    fig.suptitle(title1, fontsize=16)

    if augmented_images is not None:
        fig2 = plt.figure(2, figsize=(augmented_images.shape[0], augmented_images.shape[0]))
        for i in range(augmented_images.shape[0]):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(augmented_images[i], cmap=plt.cm.binary)
            if labels is not None:
                plt.xlabel(class_names[int(labels[i])])
        fig2.suptitle(title2, fontsize=16)
    plt.show()


def split_indexes(n_classes, n_labeled_per_class, n_validation, labels):
    labels = np.array(labels)
    train_labeled_indexes = []
    train_unlabeled_indexes = []
    validation_indexes = []

    for i in range(n_classes):
        indexes = np.where(labels == i)[0]
        np.random.shuffle(indexes)

        train_labeled_indexes.extend(indexes[:n_labeled_per_class])
        train_unlabeled_indexes.extend(indexes[n_labeled_per_class:-n_validation])
        validation_indexes.extend(indexes[-n_validation:])

    np.random.shuffle(train_unlabeled_indexes)
    np.random.shuffle(train_labeled_indexes)
    np.random.shuffle(validation_indexes)

    return train_labeled_indexes, train_unlabeled_indexes, validation_indexes


def to_tensor_dim(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])


def normalise(X):
    mean = np.mean(X, axis=(0, 1, 2))
    std = np.std(X, axis=(0, 1, 2))
    X, mean, std = [np.array(a, np.float32) for a in (X, mean, std)]
    X -= mean
    X *= 1.0 / std
    return X


def normalise2(X):
    x = X.copy()
    mean = np.mean(x, axis=(0, 1, 2)) / 255
    std = np.std(x, axis=(0, 1, 2)) / 255
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def random_flip(x):
    if np.random.rand() < 0.6:
        x = x[:, ::-1, :]
    return x.copy()


def pad(x, border=4):
    return np.pad(x, [(border, border), (border, border), (0, 0)], mode='reflect')


def pad_and_crop(x, output_size=(32, 32)):
    x = pad(x, 4)
    h, w = x.shape[:-1]
    new_h, new_w = output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    x = x[top: top + new_h, left: left + new_w, :]

    return x


def augment(X, K=1):
    X_augmented = np.tile(X, (K, 1, 1, 1))
    for i in range(X_augmented.shape[0]):
        x = X_augmented[i, :]
        x = pad_and_crop(x)
        X_augmented[i, :] = random_flip(x)
    return X_augmented


def load_and_augment_data(dataset_name, model_params):
    """
    From datasets.CIFAR10:
        dataset.data: the image as numpy array, shape: (50000, 32, 32, 3)
        dataset.targets: labels of the images as list, len: 50000
    :return:
        augmented_labeled_X: the tensor of augmented labeled images (K=1),
                             size: (n_labeled_per_class * n_classes , 32, 32, 3)
        augmented_unlabeled_X: the tensor of augmented unlabeled images (K=2),
                             size: ((N/10 - n_labeled_per_class - n_validation) * n_classes * K , 32, 32, 3)
        train_labeled_targets: the tensor of labeled targets,
                             size = n_labeled_per_class * n_classes
        train_unlabeled_targets: the tensor of unlabeled targets,
                             size = (N/10 - n_labeled_per_class - n_validation) * n_classes
    """

    # Step 1: Set the model's hyperparameters
    n_classes = model_params["n_classes"]
    n_labeled_per_class = model_params["n_labeled_per_class"]
    n_validation = model_params["n_validation"]
    K = model_params["K"]

    # Step 2: Load the dataset
    if dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(root="./datasets", train=True, download=True)
    elif dataset_name == 'SLT10':
        dataset = datasets.STL10(root="./datasets", download=True)
    else:
        raise ValueError("Invalid dataset name")

    # Step 3: Split the indexes
    train_labeled_indexes, train_unlabeled_indexes, validation_indexes = \
        split_indexes(n_classes, n_labeled_per_class, n_validation, dataset.targets)

    # Step 4: Attract the images for training, validation
    train_labeled_images = np.take(dataset.data, train_labeled_indexes, axis=0)
    train_unlabeled_images = np.take(dataset.data, train_unlabeled_indexes, axis=0)
    target_array = np.asarray(dataset.targets)
    train_labeled_targets = np.take(target_array, train_labeled_indexes, axis=0)
    train_unlabeled_targets = np.take(target_array, train_unlabeled_indexes, axis=0)
    validation_images = np.take(dataset.data, validation_indexes, axis=0)
    validation_targets = np.take(target_array, validation_indexes, axis=0)

    # Step 5: Normalise the datasets
    train_labeled_images = normalise(train_labeled_images)
    train_unlabeled_images = normalise(train_unlabeled_images)

    # Step 6: Augment training images
    augmented_labeled_X = augment(train_labeled_images, K=1)
    augmented_unlabeled_X = augment(train_unlabeled_images, K=K)

    # Take a look at some of the augmented images
    # displayImages(train_labeled_images[:10], title1="Original-Labeled", title2="Augmented-Labeled",
    #               augmented_images=augmented_labeled_X[:10], labels=train_labeled_targets[:10])
    # n_unlabeled = train_unlabeled_images.shape[0]
    # displayImages(train_unlabeled_images[:10], title1="Original-Unlabeled", title2="Augmented-Unlabeled",
    #               augmented_images=augmented_unlabeled_X[:10], labels=train_unlabeled_targets[:10])
    # displayImages(augmented_unlabeled_X[:10], title1="Augmented-Unlabeled1", title2="Augmented-Unlabeled2",
    #               augmented_images=augmented_unlabeled_X[n_unlabeled:10+n_unlabeled],
    #               labels=train_unlabeled_targets[:10])

    # Step 7: Change the dimension of np.array in oder for it to work with torch
    augmented_labeled_X = to_tensor_dim(augmented_labeled_X)
    augmented_unlabeled_X = to_tensor_dim(augmented_unlabeled_X)
    validation_images = to_tensor_dim(validation_images)

    return torch.from_numpy(augmented_labeled_X), torch.from_numpy(augmented_unlabeled_X), \
           torch.from_numpy(train_labeled_targets), torch.from_numpy(train_unlabeled_targets), \
           torch.from_numpy(validation_images), torch.from_numpy(validation_targets)
