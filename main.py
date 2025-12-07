from pathlib import Path
import torch
from torch import nn, optim

from dataset import (
    DATASET_ROOT,
    create_datasets,
    create_dataloaders,
)
from model import build_model, count_trainable_parameters
from train import train_one_epoch, evaluate
from plot import plot_metrics
from logger import Logger
import sys
from datetime import datetime


log_dir = "logs"
Path(log_dir).mkdir(exist_ok=True)

sys.stdout = Logger(
    f"{log_dir}/train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)


def main():
    #Подготовка данных
    print("Dataset root:", DATASET_ROOT)

    train_ds, val_ds, test_ds = create_datasets(DATASET_ROOT)

    print("Train images:", len(train_ds))
    print("Val images:", len(val_ds))
    print("Test images:", len(test_ds))
    print("Number of classes:", len(train_ds.classes))
    print("Classes:", train_ds.classes)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=32
    )

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    # Модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #обучаем кто богатый на CUDA, у кого нет -- на CPU
    print("Device:", device)

    num_classes = len(train_ds.classes)
    model = build_model(num_classes).to(device)

    print(model)
    print("Trainable parameters:", count_trainable_parameters(model))

    #Обучение модели
    criterion = nn.CrossEntropyLoss() #Функция потерь для каждого батча наказывает за большие уверенности -- выше Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4) #меняем веса если лосс есть, Adam -- подход при котором при ходе градиента он сам считает шаг и среднее скользящее а не просто минус градиент (на курсе я именно так делал)

    num_epochs = 5 #чтобы наверняка все запомнила 

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate( #оценка, здесь веса не трогаем
            model,
            val_loader,
            criterion,
            device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}"
        )


    # Финальная оценка
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(f"\nTest accuracy: {test_acc:.4f}")
    # сохраняем обученную модель
    torch.save(model.state_dict(), "resnet_grocery.pth")
    print("Model saved to resnet_grocery.pth")


    # Графики
    plot_metrics(history)


if __name__ == "__main__":
    main()
