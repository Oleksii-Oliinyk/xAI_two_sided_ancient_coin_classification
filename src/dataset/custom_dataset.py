from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.constants import IMAGE_EXTENSIONS, SIDES


IMAGE_SIZE = 224
# ImageNet stats, grayscale is repeated to 3 channels so pretrained weights still fit
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_base_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_augment_transform():
    # no flips: mirrored coins (text, portraits) don't exist in reality
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class CoinDataset(Dataset):

    def __init__(self, split_root, side, augment=False, augment_factor=3):
        """
        split_root:     e.g. dataset/data/train, contains 'Observe'/'Reverse' with one folder per class
        side:           'Observe' or 'Reverse', only images of this side are loaded
        augment:        if True every image appears `augment_factor` times:
                        once original + (augment_factor - 1) randomly augmented copies
        """
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got '{side}'")

        self.split_root = Path(split_root)
        self.side = side
        self.side_root = self.split_root / side
        self.augment = augment
        self.augment_factor = augment_factor if augment else 1

        self.base_transform = get_base_transform()
        self.augment_transform = get_augment_transform()

        # class folders are numbers "1".."60" -> sort numerically so label i == class folder i+1
        class_names = [d.name for d in self.side_root.iterdir() if d.is_dir()]
        self.classes = sorted(class_names, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else c))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for class_dir in sorted(self.side_root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = self.class_to_idx[class_dir.name]
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((f, label))

    def __len__(self):
        return len(self.samples) * self.augment_factor

    def __getitem__(self, idx):
        path, label = self.samples[idx // self.augment_factor]
        image = Image.open(path).convert("RGB")

        is_copy = idx % self.augment_factor != 0
        transform = self.augment_transform if is_copy else self.base_transform
        return transform(image), label
