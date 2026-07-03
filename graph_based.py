import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel
import networkx as nx
from scipy.sparse import csgraph
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

num_labeled = 100
labeled_indices = np.random.choice(len(full_train), num_labeled, replace=False)
unlabeled_indices = list(set(range(len(full_train))) - set(labeled_indices))

labeled_set = Subset(full_train, labeled_indices)
unlabeled_set = Subset(full_train, unlabeled_indices)

labeled_loader = DataLoader(labeled_set, batch_size=32, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

class GraphSemiSupervisedNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super(GraphSemiSupervisedNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.encoder(x)
        output = self.classifier(features)
        return output, features

class GraphSemiSupervisedLearning:
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, alpha=0.1, sigma=1.0):
        self.model = GraphSemiSupervisedNN(input_dim, hidden_dim, output_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.sigma = sigma
        self.train_losses = []
        self.train_accs = []
        self.test_accs = []
        
    def compute_graph_laplacian(self, features):
        features_np = features.detach().cpu().numpy()
        W = rbf_kernel(features_np, gamma=1.0/(2*self.sigma**2))
        np.fill_diagonal(W, 0)
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        L_sym = csgraph.laplacian(W, normed=True)
        return torch.FloatTensor(L_sym).to(device)
    
    def graph_regularization_loss(self, features):
        n = features.size(0)
        L = self.compute_graph_laplacian(features)
        graph_loss = torch.trace(torch.mm(torch.mm(features.T, L), features)) / (n * n)
        return graph_loss
    
    def train_epoch(self, labeled_loader, unlabeled_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        num_batches = max(len(labeled_loader), len(unlabeled_loader))
        
        for batch_idx in range(num_batches):
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
            
            combined_data = torch.cat([labeled_data, unlabeled_data], dim=0).to(device)
            combined_labels = torch.cat([labeled_labels.to(device), 
                                        torch.full((unlabeled_data.size(0),), -1, device=device)])
            
            self.optimizer.zero_grad()
            
            outputs, features = self.model(combined_data)
            
            labeled_mask = combined_labels != -1
            supervised_loss = self.criterion(outputs[labeled_mask], combined_labels[labeled_mask].long())
            
            graph_loss = self.graph_regularization_loss(features)
            
            total_batch_loss = supervised_loss + self.alpha * graph_loss
            
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            with torch.no_grad():
                predicted = torch.argmax(outputs[labeled_mask][:, :10], dim=1)
                correct += (predicted == combined_labels[labeled_mask]).sum().item()
                total += labeled_mask.sum().item()
        
        avg_loss = total_loss / num_batches
        accuracy = 100 * correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self, labeled_loader, unlabeled_loader, epochs=50):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(labeled_loader, unlabeled_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            test_acc = self.evaluate(test_loader)
            self.test_accs.append(test_acc)
            
            print(f'Epoch [{epoch+1}/{epochs}] Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs, _ = self.model(data)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total
    
    def visualize_embeddings(self, test_loader, method='tsne'):
        self.model.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                _, features = self.model(data)
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())
        
        features = np.vstack(features_list)
        labels = np.hstack(labels_list)
        
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        
        features_2d = reducer.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'Embedding Visualization ({method.upper()})')
        plt.show()
    
    def plot_learning_curves(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.train_accs, label='Train Accuracy')
        ax2.plot(self.test_accs, label='Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

model = GraphSemiSupervisedLearning(input_dim=784, hidden_dim=256, output_dim=10, alpha=0.1, sigma=1.0)

model.train(labeled_loader, unlabeled_loader, epochs=50)

final_test_acc = model.evaluate(test_loader)
print(f'Final Test Accuracy: {final_test_acc:.2f}%')

model.visualize_embeddings(test_loader, method='tsne')

model.plot_learning_curves()

print(f"Graph-based Semi-Supervised Learning completed with {num_labeled} labeled samples")
print(f"Final Test Accuracy: {final_test_acc:.2f}%")
