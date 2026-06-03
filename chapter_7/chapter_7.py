import os
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


data_root = "./data/helodrys_nebula_images/Nebulae"

TRAIN_CLASSES = [
    "Dark Nebula",
    "Emission Nebula",
    "Planetary Nebula",
    "Reflection Nebula",
    "Supernova Remnants"
]

VAL_RAW_CLASSES = TRAIN_CLASSES
CLASS_MAPPING = {}

num_epochs = 25
lr = 0.001
batch_size = 32
print_every = 10
num_workers = 4

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 is_valid_file=None, valid_classes=None, class_mapping=None):
        self.valid_classes = valid_classes
        self.class_mapping = class_mapping or {}
        super().__init__(root, transform=transform,
                         target_transform=target_transform,
                         is_valid_file=is_valid_file)

    def find_classes(self, directory):
        existing_raw = [subdir for subdir in os.listdir(directory)
                        if os.path.isdir(os.path.join(directory, subdir))
                        and subdir in self.valid_classes]
        mapped = [self.class_mapping.get(cls, cls) for cls in existing_raw]

        valid_mapped = []
        for raw_cls, mapped_cls in zip(existing_raw, mapped):
            cls_path = os.path.join(directory, raw_cls)
            try:
                has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg'))
                                 for f in os.listdir(cls_path))
            except (PermissionError, FileNotFoundError):
                has_images = False

            if has_images and mapped_cls in TRAIN_CLASSES:
                valid_mapped.append(mapped_cls)

        valid_mapped = list(set(valid_mapped))
        valid_mapped.sort(key=lambda x: TRAIN_CLASSES.index(x))
        return valid_mapped, {cls: i for i, cls in enumerate(valid_mapped)}


class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        sample_index = self.indices[idx]
        path, target = self.dataset.samples[sample_index]
        img = self.dataset.loader(path)
        if self.transform:
            img = self.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target


def is_valid_file(path, valid_classes):
    parent_dir = os.path.basename(os.path.dirname(path))
    return parent_dir in valid_classes


print(f"正在加载训练集，根目录: {data_root}")
train_dataset = CustomImageFolder(
    root=data_root,
    transform=train_transform,
    is_valid_file=lambda x: is_valid_file(x, TRAIN_CLASSES),
    valid_classes=TRAIN_CLASSES,
    class_mapping=CLASS_MAPPING
)

if len(train_dataset) == 0:
    raise FileNotFoundError(f"训练集无有效样本！检查：{TRAIN_CLASSES}")
print(f"训练集加载完成，样本数: {len(train_dataset)}")

val_dir = os.path.join(data_root, "Validation")
val_dataset = None
if os.path.exists(val_dir):
    print(f"发现独立验证集目录: {val_dir}")
    val_dataset = CustomImageFolder(
        root=val_dir,
        transform=val_transform,
        is_valid_file=lambda x: is_valid_file(x, VAL_RAW_CLASSES),
        valid_classes=VAL_RAW_CLASSES,
        class_mapping=CLASS_MAPPING
    )

    if val_dataset and set(val_dataset.classes) != set(TRAIN_CLASSES):
        print(f"警告: 验证集类别不匹配，将拆分训练集 → 训练集:{TRAIN_CLASSES} 验证集:{val_dataset.classes}")
        val_dataset = None
    if val_dataset and len(val_dataset) == 0:
        print("警告: Validation目录无有效样本，将拆分训练集")
        val_dataset = None
else:
    print("未发现独立验证集目录，将拆分训练集")

if val_dataset is None:
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    train_dataset = TransformSubset(train_subset.dataset, train_subset.indices, train_transform)
    val_dataset = TransformSubset(val_subset.dataset, val_subset.indices, val_transform)
    print(f"训练集已拆分为: 训练 {train_size} 样本, 验证 {val_size} 样本")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


class HybridSequential(nn.Sequential):
    def add(self, *blocks):
        for block in blocks:
            self.add_module(str(len(self)), block)
        return self


def Conv2D(channels, kernel_size, strides=1, padding=0):
    return nn.LazyConv2d(channels, kernel_size=kernel_size, stride=strides, padding=padding, bias=False)


def BatchNorm(num_features):
    return nn.BatchNorm2d(num_features)


class Activation(nn.Module):
    def __init__(self, act_type):
        super().__init__()
        if act_type != 'relu':
            raise ValueError(f"不支持的激活函数: {act_type}")
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x)


class GlobalAvgPool2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return self.pool(x)


class Dense(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.linear = nn.LazyLinear(units)

    def forward(self, x):
        return self.linear(torch.flatten(x, 1))


class Residual(nn.Module):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides, bias=False)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides, bias=False) if use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return self.relu(y + x)


def resnet18(num_classes):
    net = HybridSequential()
    net.add(Conv2D(64, kernel_size=3, strides=1, padding=1),
            BatchNorm(64),
            Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(GlobalAvgPool2D(),
            Dense(num_classes))
    return net


num_classes = len(TRAIN_CLASSES)
print(f"使用函数定义 ResNet18，类别数: {num_classes}")

model = resnet18(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
with torch.no_grad():
    model(torch.zeros(1, 3, 224, 224, device=device))
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0


for epoch in range(num_epochs):
    print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    train_iter = tqdm(train_loader, desc="Training")
    for batch_idx, (inputs, labels) in enumerate(train_iter):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()
        if (batch_idx + 1) % print_every == 0:
            batch_acc = (preds == labels).float().mean().item()
            train_iter.set_postfix(batch_loss=f"{loss.item():.4f}", batch_acc=f"{batch_acc:.4f}")

    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / train_total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    print(f"训练结果: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    val_iter = tqdm(val_loader, desc="Validation")
    with torch.no_grad():
        for inputs, labels in val_iter:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    print(f"验证结果: Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"模型保存（验证准确率: {val_acc:.4f}）")
