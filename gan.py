import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

num_labeled = 1000
labeled_indices = np.random.choice(len(full_train), num_labeled, replace=False)
unlabeled_indices = list(set(range(len(full_train))) - set(labeled_indices))

labeled_set = Subset(full_train, labeled_indices)
unlabeled_set = Subset(full_train, unlabeled_indices)

labeled_loader = DataLoader(labeled_set, batch_size=64, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, True),
        )
        self.classifier = nn.Linear(128, num_classes + 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.features(x)
        output = self.classifier(features)
        return output, features

class SemiSupervisedGAN:
    def __init__(self, latent_dim=100, num_classes=10):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.generator = Generator(latent_dim).to(device)
        self.discriminator = Discriminator(num_classes).to(device)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.criterion = nn.CrossEntropyLoss()
        self.g_losses = []
        self.d_losses = []
        self.accuracies = []
        
    def train(self, labeled_loader, unlabeled_loader, epochs=50):
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(epochs):
            labeled_iter = iter(labeled_loader)
            unlabeled_iter = iter(unlabeled_loader)
            
            total_batches = max(len(labeled_loader), len(unlabeled_loader))
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_acc = 0
            batch_count = 0
            
            for batch_idx in range(total_batches):
                try:
                    labeled_data, labeled_labels = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    labeled_data, labeled_labels = next(labeled_iter)
                
                try:
                    unlabeled_data, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_data, _ = next(unlabeled_iter)
                
                batch_size = labeled_data.size(0)
                unlabeled_batch_size = unlabeled_data.size(0)
                
                real_data = torch.cat([labeled_data, unlabeled_data], dim=0).to(device)
                real_labels = torch.cat([
                    labeled_labels.to(device),
                    torch.full((unlabeled_batch_size,), self.num_classes, device=device)
                ], dim=0)
                
                z = torch.randn(batch_size + unlabeled_batch_size, self.latent_dim, device=device)
                fake_data = self.generator(z).detach()
                fake_labels = torch.full((batch_size + unlabeled_batch_size,), self.num_classes, device=device)
                
                all_data = torch.cat([real_data, fake_data], dim=0)
                all_labels = torch.cat([real_labels, fake_labels], dim=0)
                
                self.d_optimizer.zero_grad()
                outputs, _ = self.discriminator(all_data)
                d_loss = self.criterion(outputs, all_labels)
                d_loss.backward()
                self.d_optimizer.step()
                
                z = torch.randn(batch_size + unlabeled_batch_size, self.latent_dim, device=device)
                fake_data = self.generator(z)
                fake_labels = torch.full((batch_size + unlabeled_batch_size,), self.num_classes, device=device)
                
                self.g_optimizer.zero_grad()
                outputs, _ = self.discriminator(fake_data)
                g_loss = self.criterion(outputs, fake_labels)
                g_loss.backward()
                self.g_optimizer.step()
                
                self.discriminator.eval()
                with torch.no_grad():
                    labeled_outputs, _ = self.discriminator(labeled_data.to(device))
                    predicted = torch.argmax(labeled_outputs[:, :self.num_classes], dim=1)
                    acc = (predicted == labeled_labels.to(device)).float().mean().item()
                self.discriminator.train()
                
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_acc += acc
                batch_count += 1
            
            avg_g_loss = epoch_g_loss / batch_count
            avg_d_loss = epoch_d_loss / batch_count
            avg_acc = epoch_acc / batch_count
            
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            self.accuracies.append(avg_acc)
            
            print(f'Epoch [{epoch+1}/{epochs}] G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}, Acc: {avg_acc:.4f}')
    
    def evaluate(self, test_loader):
        self.discriminator.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs, _ = self.discriminator(data)
                predicted = torch.argmax(outputs[:, :self.num_classes], dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy
    
    def visualize_latent_space(self, test_loader, method='tsne'):
        self.discriminator.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                _, features = self.discriminator(data)
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())
        
        features = np.vstack(features_list)
        labels = np.hstack(labels_list)
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
        
        features_2d = reducer.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'Latent Space Visualization ({method.upper()})')
        plt.show()
    
    def generate_samples(self, num_samples=16):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.generator(z)
        
        samples = (samples + 1) / 2
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                ax.imshow(samples[i].cpu().squeeze(), cmap='gray')
            ax.axis('off')
        plt.suptitle('Generated Samples')
        plt.show()

gan = SemiSupervisedGAN(latent_dim=100, num_classes=10)

gan.train(labeled_loader, unlabeled_loader, epochs=30)

test_accuracy = gan.evaluate(test_loader)

gan.visualize_latent_space(test_loader, method='tsne')

gan.generate_samples(16)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(gan.g_losses, label='Generator Loss')
plt.plot(gan.d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')

plt.subplot(1, 3, 2)
plt.plot(gan.accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Supervised Accuracy')

plt.tight_layout()
plt.show()

print(f"Final Test Accuracy: {test_accuracy:.2f}%")
