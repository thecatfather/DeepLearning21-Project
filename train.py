import data


model_params = {
    "n_classes": 10,
    "n_labeled_per_class": 3000,
    "n_validation": 500,
    "K": 2
}
augmented_labeled_X, augmented_unlabeled_X, train_labeled_targets, train_unlabeled_targets, \
    validation_images, validation_targets = data.load_and_augment_data('CIFAR10', model_params)

print(augmented_labeled_X.size())
print(augmented_unlabeled_X.size())
print(train_labeled_targets.size())
print(train_unlabeled_targets.size())
