from torchvision import transforms, datasets
import cupy as cp

# Data Loading and Preprocessing
def load_cifar10(binary=False, target_class=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    if binary:
        # Binary classification (airplane vs rest)
        train_set.targets = [0 if t == target_class else 1 for t in train_set.targets]
        test_set.targets = [0 if t == target_class else 1 for t in test_set.targets]

    # Convert to CuPy arrays
    x_train = cp.array([x.numpy() for x, _ in train_set]).reshape(len(train_set), -1) # Flattening samples
    y_train = cp.array(train_set.targets)
    x_test = cp.array([x.numpy() for x, _ in test_set]).reshape(len(test_set), -1) # Flattening samples
    y_test = cp.array(test_set.targets)

    return x_train, y_train, x_test, y_test