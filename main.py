# @kevserbusrayildirim

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")


if __name__ == '__main__':
    # CIFAR-10 veri setini yükleme ve normalizasyon işlemleri
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # MLP modeli oluşturma
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            # Giriş katmanı
            self.fc1 = nn.Linear(32*32*3, 1000)
            self.bn1 = nn.BatchNorm1d(1000)
            self.relu1 = nn.ReLU()

            # Gizli katman 1
            self.fc2 = nn.Linear(1000, 512)
            self.bn2 = nn.BatchNorm1d(512)
            self.relu2 = nn.ReLU()

            # Gizli katman 2
            self.fc3 = nn.Linear(512, 256)
            self.bn3 = nn.BatchNorm1d(256)
            self.relu3 = nn.ReLU()

            # Çıkış katmanı
            self.fc4 = nn.Linear(256, 10)  # 10 sınıflı bir çıkış

        def forward(self, x):
            # Giriş verisini düzleştirme
            x = x.view(-1, 32*32*3)

            # 1. Gizli katman ve Batch Normalization
            x = self.bn1(self.relu1(self.fc1(x)))

            # 2. Gizli katman ve Batch Normalization
            x = self.bn2(self.relu2(self.fc2(x)))

            # 3. Gizli katman ve Batch Normalization
            x = self.bn3(self.relu3(self.fc3(x)))

            # Çıkış katmanı
            x = self.fc4(x)
            return x

    # Modeli oluşturma
    model = MLP()
    model.to(mps_device)

    # Optimizasyon algoritmaları ve öğrenme oranları
    optimizers = {
        'RMSprop with lr=1e-2': optim.RMSprop(model.parameters(), lr=1e-2, weight_decay=1e-5),
        'RMSprop with lr=1e-6': optim.RMSprop(model.parameters(), lr=1e-6, weight_decay=1e-5),
        'SGD with lr=1e-2': optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-5),
        'SGD with lr=1e-6': optim.SGD(model.parameters(), lr=1e-6, weight_decay=1e-5),
        'Adam with lr=1e-2': optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5),
        'Adam with lr=1e-6': optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-5),
        'Adagrad with lr=1e-2': optim.Adagrad(model.parameters(), lr=1e-2, weight_decay=1e-5),
        'Adagrad with lr=1e-6': optim.Adagrad(model.parameters(), lr=1e-6, weight_decay=1e-5)
    }

    # Loss fonksiyonu
    criterion = nn.CrossEntropyLoss()

    # Epoch sayısı
    num_epochs = 100


    # Eğitim ve değerlendirme fonksiyonları
    def train(model, optimizer, criterion, trainloader):
        # Eğitim modu
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(mps_device), labels.to(mps_device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return running_loss / len(trainloader), correct / total


    def validate(model, criterion, dataloader):
        # Değerlendirme modu
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(mps_device), labels.to(mps_device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return running_loss / len(dataloader), correct / total


    # Deneylerin yapılması
    for optimizer_name, optimizer in optimizers.items():
        print(f"Optimizer: {optimizer_name}")
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
        for epoch in range(num_epochs):
            train_loss, train_accuracy = train(model, optimizer, criterion, trainloader)
            val_loss, val_accuracy = validate(model, criterion, testloader)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1}/{num_epochs} => "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} | "
                  f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Grafiklerin çizilmesi
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        current_directory = os.getcwd()  # Şu anki çalışma dizini
        graph_filename = os.path.join(current_directory, f"{optimizer_name.replace(' ', '_')}_graph.png")
        plt.savefig(graph_filename)
        plt.close()

        # Confusion Matrix oluşturulması
        # Confusion Matrix oluşturulması
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(mps_device), labels.to(mps_device)  # Veriyi ve etiketleri MPS cihazına taşı
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                all_labels.extend(labels.cpu().numpy())  # labels zaten CPU'da olduğu için taşıma işlemine gerek yok
                all_preds.extend(predicted.cpu().numpy())

        # Confusion Matrix çizme
        cm = confusion_matrix(all_labels, all_preds)
        print(f"Confusion Matrix:\n{cm}\n\n")
        sns.set(font_scale=1.2)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=.5, xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted Classes")
        plt.ylabel("Actual Classes")
        plt.title(f"{optimizer_name} Confusion Matrix Heatmap")
        cm_filename = f"{optimizer_name.replace(' ', '_')}_cm.png"
        plt.savefig(cm_filename)
        plt.close()

