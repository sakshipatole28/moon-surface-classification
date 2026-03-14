import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ModelTrainer:

    def __init__(self, model, train_loader, val_loader, epochs, lr, num_classes, model_save_path):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.num_classes = num_classes
        self.model_save_path = model_save_path

    def train(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)

        # ---------- compute class weights ----------
        labels_list = []

        for _, labels in self.train_loader:
            labels_list.extend(labels.numpy())

        labels_array = np.array(labels_list)

        class_counts = np.bincount(labels_array, minlength=self.num_classes)

        class_counts[class_counts == 0] = 1

        weights = 1.0 / class_counts
        weights = weights / weights.sum()

        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):

            self.model.train()

            running_loss = 0

            for images, labels in self.train_loader:

                images = images.to(device)
                labels = labels.long().to(device)

                optimizer.zero_grad()

                outputs = self.model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)

            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f}")

            self.validate(device)

        torch.save(self.model.state_dict(), self.model_save_path)

        print("Model saved successfully")

    def validate(self, device):

        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in self.val_loader:

                images = images.to(device)
                labels = labels.long().to(device)

                outputs = self.model(images)

                _, predicted = torch.max(outputs, 1)

                correct += (predicted == labels).sum().item()

                total += labels.size(0)

        accuracy = correct / total

        print(f"Validation Accuracy: {accuracy:.4f}")
