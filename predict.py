from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import build_model
from logger import Logger
import sys
from datetime import datetime
from pathlib import Path

log_dir = "logs"
Path(log_dir).mkdir(exist_ok=True)

sys.stdout = Logger(
    f"{log_dir}/predict_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

# классы (в том же порядке, что ImageFolder)
CLASSES = ["Fruit", "Packages", "Vegetables"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weights_path: str):
    model = build_model(num_classes=len(CLASSES))
    model.load_state_dict(#загружаем веса
        torch.load(weights_path, map_location=DEVICE,  weights_only=True)) #загружаем модель
    model.to(DEVICE)
    model.eval()
    return model


def get_transform(): #превращаем входную картинку в тензор
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_image(model, image_path: str):
    image = Image.open(image_path).convert("RGB")
    transform = get_transform()
    tensor = transform(image).unsqueeze(0).to(DEVICE) #превращаем картинку в тензор + добавляем размер батча одна картинка 

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1) #убираем отрицательные, сумма по классам 1 (чтоб все признаки в единицу помещались) (0,01 0,96 0,03) -- вот и расчет вероятности какой класс картинка
        confidence, pred_idx = torch.max(probs, dim=1) #индекс класса + уверенность из софтмакса

    return CLASSES[pred_idx.item()], confidence.item()


if __name__ == "__main__":
    model = load_model("resnet_grocery.pth")

    image_path = "test_image.jpg"  # ← заменишь на свою картинку
    label, confidence = predict_image(model, image_path)

    print("Predicted class:", label)
    print("Confidence:", round(confidence, 3))
