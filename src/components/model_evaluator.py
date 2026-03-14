import torch
from sklearn.metrics import accuracy_score


class ModelEvaluator:

    def __init__(self, model, val_loader):

        self.model = model
        self.val_loader = val_loader

    def evaluate(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)

        self.model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():

            for images, labels in self.val_loader:

                images = images.to(device)

                outputs = self.model(images)

                _, preds = torch.max(outputs, 1)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)

        print(f"Final Validation Accuracy: {acc}")
