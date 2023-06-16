import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


_DATASETS = ['rn50x16openai', 'vitb16laion400m', 'vitb16openai', 'imagenet', 'cifar10', 'cifar100', 'mnist', 'imagenette', 'imagewoof']
_IMAGENET_RGB_MEANS = (0.485, 0.456, 0.406)
_IMAGENET_RGB_STDS = (0.229, 0.224, 0.225)
_CIFAR10_RGB_MEANS = (0.491, 0.482, 0.447)
_CIFAR10_RGB_STDS = (0.247, 0.243, 0.262)
_CIFAR100_RGB_MEANS = (0.507, 0.487, 0.441)
_CIFAR100_RGB_STDS = (0.267, 0.256, 0.276)
_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        assert data is not None, "data cannot be None!"
        assert targets is not None, "targets cannot be None!"
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    @property
    def rows(self):
        return self.data.shape[0]

    @property
    def cols(self):
        return self.data.shape[1]


def get_cifar10_datasets(data_dir):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR10_RGB_MEANS, std=_CIFAR10_RGB_STDS)
    ])
    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR10_RGB_MEANS, std=_CIFAR10_RGB_STDS)
    ])
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset


def get_cifar100_datasets(data_dir):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR100_RGB_MEANS, std=_CIFAR100_RGB_STDS)
    ])
    train_dataset = datasets.CIFAR100(root=data_dir, train=True,
                                      download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR100_RGB_MEANS, std=_CIFAR100_RGB_STDS)
    ])
    test_dataset = datasets.CIFAR100(root=data_dir, train=False,
                                     download=True, transform=test_transform)

    return train_dataset, test_dataset


def get_mnist_datasets(data_dir):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_MNIST_MEAN, std=_MNIST_STD)
    ])
    train_dataset = datasets.MNIST(root=data_dir, train=True,
                                   download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_MNIST_MEAN, std=_MNIST_STD)
    ])
    test_dataset = datasets.MNIST(root=data_dir, train=False,
                                  transform=test_transform)

    return train_dataset, test_dataset


def get_imagenet_datasets(data_dir, img_size=224):
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS),
    ])
    train_dir = os.path.join(os.path.expanduser(data_dir), 'train')
    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_dir = os.path.join(os.path.expanduser(data_dir), 'val')
    non_rand_resize_scale = 256.0 / img_size  # standard
    test_transform = transforms.Compose([
            transforms.Resize(round(non_rand_resize_scale * img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS),
        ])
    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset


def get_imagenette_datasets(data_dir, img_size=224):
    if 'imagenette' not in data_dir.lower():
        raise RuntimeError('Not a valid path for ImageNette dataset! It should contain "imagenette".')
    return get_imagenet_datasets(data_dir, img_size)


def get_imagewoof_datasets(data_dir, img_size=224):
    if 'imagewoof' not in data_dir.lower():
        raise RuntimeError('Not a valid path for ImageWoof dataset! It should contain "imagewoof".')
    return get_imagenet_datasets(data_dir, img_size)


def get_custom_datasets(path, suffix=''):
    x_train = torch.load(os.path.join(path, f'features_train{suffix}.pt'))
    y_train = torch.load(os.path.join(path, f'targets_train{suffix}.pt'))
    x_val = torch.load(os.path.join(path, f'features_val{suffix}.pt'))
    y_val = torch.load(os.path.join(path, f'targets_val{suffix}.pt'))
    data_train = CustomDataset(data=x_train, targets=y_train)
    data_val = CustomDataset(data=x_val, targets=y_val)
    return data_train, data_val


def get_rn50x16openai_datasets(path):
    return get_custom_datasets(path)


def get_vitb16laion400m_datasets(path):
    return get_custom_datasets(path)


def get_vitb16openai_datasets(path):
    return get_custom_datasets(path, suffix='_sparsity=0.000')


def get_datasets(dataset, dataset_dir):
    """Creates a tuple: (train_dataset, test_dataset)"""
    assert dataset in _DATASETS, f"Unexpected value {dataset} for a dataset. Supported: {_DATASETS}"
    return globals()[f"get_{dataset}_datasets"](dataset_dir)
