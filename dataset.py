from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Путь к корню датасета относительно данного файла
DATASET_ROOT = Path(__file__).parent / "GroceryStoreDataset" / "dataset"


def get_transforms(train: bool):

    #функция редактирует картинки: в случае, если речь идет о трейн выборке, то мы берем изображение и различными образами изменяем его (режем картинку, переворачиваем, меняем освещени
    #е, чтобы лучше училась)
    #Далее картинку форматируем в формат PyTroch
    #Далее задаем картинки параметры, на параметрах картинок которых училась модель
    #А для этапа валидации и теста просто подбиваем картинки под один формат, чтобы она уже могла оценивать  
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_datasets(dataset_root: Path = DATASET_ROOT):
 #Создаем три выборки, указывая путь из датасета
    train_dataset = datasets.ImageFolder(
        root=dataset_root / "train",
        transform=get_transforms(train=True)
    )

    val_dataset = datasets.ImageFolder(
        root=dataset_root / "val",
        transform=get_transforms(train=False)
    )

    test_dataset = datasets.ImageFolder(
        root=dataset_root / "test",
        transform=get_transforms(train=False)
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 32 #выборка из выборки на 32 картинки
):
#ДатаЛоадеры -- процесс, как будут подаваться данные в модель. Устанавливают так, чтоб при трейне перемешивать -- для обучения
#Для вала и теста не мешаем, потому что таким образом непонятно как оценивать качество между итерациями (эпохами)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
