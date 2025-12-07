import torch
from torch import nn
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train() #Включаем режим обучения, чтоб сказать слою BatchNorm2D что щас обучение
    #просто счетчики
    running_loss = 0.0
    correct = 0
    total = 0
    #запускаем цикл по батчам
    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device) #обучаем данные на GPU компа (чтоб на CPU не переходили вычисления)
        labels = labels.to(device)

        optimizer.zero_grad() #обнуляем градиенты предыдущего обучения чтоб случайно градиент предыдущего обучения не попался в новой
        outputs = model(images) #заводим модель в РесНет
        loss = criterion(outputs, labels) #считаем ошибку
        loss.backward() #вычисляем градиент (двигает точку чтоб нашла минимум (абсолютный а не локальный))
        optimizer.step() #обновляем веса которые нашли после градиента

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    #здесь подходим итоги вычисления
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    #оцениваем качество модели при валидации и тесте
    model.eval() #переключаем модель в режим оценки (не теста)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): #выключаем поиск градиентов (нам не нужно при валидации и тесте)
        for images, labels in tqdm(dataloader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
