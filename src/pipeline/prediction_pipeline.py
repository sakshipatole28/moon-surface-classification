import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.components.model_builder import CraterCNN


class PredictionPipeline:

    def __init__(self, model_path):

        csv = pd.read_csv("HONS-Lunar-AI-1/train/_classes.csv")

        self.classes = list(csv.columns)[1:]

        num_classes = len(self.classes)

        self.model = CraterCNN(num_classes)

        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])

    def predict(self, image_path):

        image = Image.open(image_path).convert("RGB")

        img = self.transform(image).unsqueeze(0)

        with torch.no_grad():

            output = self.model(img)

            probs = torch.softmax(output, dim=1)

            confidence, pred = torch.max(probs, 1)

        prediction = self.classes[pred.item()]

        return prediction, float(confidence.item())
