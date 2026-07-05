import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.stats import multivariate_normal
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

num_labeled = 500
labeled_indices = np.random.choice(len(full_train), num_labeled, replace=False)
unlabeled_indices = list(set(range(len(full_train))) - set(labeled_indices))

labeled_set = Subset(full_train, labeled_indices)
unlabeled_set = Subset(full_train, unlabeled_indices)

labeled_loader = DataLoader(labeled_set, batch_size=64, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Linear(128 * 4 * 4, 128)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class GMMSemiSupervisedLearning:
    def __init__(self, n_components=10, n_features=128, n_classes=10):
        self.n_components = n_components
        self.n_features = n_features
        self.n_classes = n_classes
        
        self.feature_extractor = FeatureExtractor().to(device)
        self.optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        self.gmm = None
        self.classifier = None
        self.scaler = StandardScaler()
        
        self.train_losses = []
        self.train_accs = []
        self.test_accs = []
        
    def extract_features(self, dataloader):
        self.feature_extractor.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for data, label in dataloader:
                data = data.to(device)
                feat = self.feature_extractor(data)
                features.append(feat.cpu().numpy())
                labels.append(label.numpy())
        
        features = np.vstack(features)
        labels = np.hstack(labels)
        return features, labels
    
    def fit_gmm(self, unlabeled_loader, labeled_features=None, labeled_labels=None):
        print("Extracting features from unlabeled data...")
        unlabeled_features, _ = self.extract_features(unlabeled_loader)
        
        if labeled_features is not None:
            all_features = np.vstack([unlabeled_features, labeled_features])
        else:
            all_features = unlabeled_features
        
        all_features_scaled = self.scaler.fit_transform(all_features)
        
        print(f"Fitting GMM with {self.n_components} components on {len(all_features_scaled)} samples...")
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            max_iter=200,
            random_state=42,
            n_init=3
        )
        self.gmm.fit(all_features_scaled)
        
        if labeled_features is not None and labeled_labels is not None:
            labeled_features_scaled = self.scaler.transform(labeled_features)
            component_probs = self.gmm.predict_proba(labeled_features_scaled)
            
            self.classifier = np.zeros((self.n_components, self.n_classes))
            for i in range(len(labeled_labels)):
                label = labeled_labels[i]
                probs = component_probs[i]
                self.classifier[:, label] += probs
            
            self.classifier = self.classifier / (self.classifier.sum(axis=1, keepdims=True) + 1e-10)
        else:
            self.classifier = np.ones((self.n_components, self.n_classes)) / self.n_classes
        
        return self.gmm
    
    def predict_with_gmm(self, features):
        features_scaled = self.scaler.transform(features)
        component_probs = self.gmm.predict_proba(features_scaled)
        class_probs = component_probs @ self.classifier
        predictions = np.argmax(class_probs, axis=1)
        return predictions
    
    def train_epoch(self, labeled_loader, unlabeled_loader):
        self.feature_extractor.train()
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
            
            labeled_data = labeled_data.to(device)
            labeled_labels = labeled_labels.to(device)
            
            self.optimizer.zero_grad()
            
            labeled_features = self.feature_extractor(labeled_data)
            labeled_outputs = nn.Linear(128, 10).to(device)(labeled_features)
            
            loss = self.criterion(labeled_outputs, labeled_labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            predicted = torch.argmax(labeled_outputs, dim=1)
            correct += (predicted == labeled_labels).sum().item()
            total += labeled_labels.size(0)
        
        avg_loss = total_loss / num_batches
        accuracy = 100 * correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self, labeled_loader, unlabeled_loader, epochs=30):
        print("Initial feature extraction for GMM...")
        labeled_features, labeled_labels = self.extract_features(labeled_loader)
        
        print("Fitting initial GMM...")
        self.fit_gmm(unlabeled_loader, labeled_features, labeled_labels)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(labeled_loader, unlabeled_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            test_acc = self.evaluate(test_loader)
            self.test_accs.append(test_acc)
            
            if (epoch + 1) % 5 == 0:
                print(f"Refitting GMM at epoch {epoch+1}...")
                labeled_features, labeled_labels = self.extract_features(labeled_loader)
                self.fit_gmm(unlabeled_loader, labeled_features, labeled_labels)
            
            print(f'Epoch [{epoch+1}/{epochs}] Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    def evaluate(self, test_loader):
        self.feature_extractor.eval()
        
        features, labels = self.extract_features(test_loader)
        predictions = self.predict_with_gmm(features)
        
        accuracy = 100 * np.mean(predictions == labels)
        return accuracy
    
    def analyze_gmm_components(self):
        if self.gmm is None:
            print("GMM not fitted yet!")
            return
        
        print(f"GMM Components: {self.n_components}")
        print(f"Means shape: {self.gmm.means_.shape}")
        print(f"Covariances shape: {self.gmm.covariances_.shape}")
        print(f"Weights: {self.gmm.weights_[:5]}...")
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(self.n_components), self.gmm.weights_)
        plt.xlabel('Component Index')
        plt.ylabel('Weight')
        plt.title('GMM Component Weights')
        plt.show()
    
    def visualize_embeddings(self, test_loader, method='tsne'):
        self.feature_extractor.eval()
        
        features, labels = self.extract_features(test_loader)
        
        predictions = self.predict_with_gmm(features)
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=2)
        
        features_2d = reducer.fit_transform(features)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
        ax1.set_title(f'True Labels ({method.upper()})')
        plt.colorbar(scatter1, ax=ax1)
        
        scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=predictions, cmap='tab10', alpha=0.6)
        ax2.set_title(f'GMM Predictions ({method.upper()})')
        plt.colorbar(scatter2, ax=ax2)
        
        plt.tight_layout()
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
    
    def visualize_gmm_components(self, test_loader):
        self.feature_extractor.eval()
        
        features, labels = self.extract_features(test_loader)
        features_scaled = self.scaler.transform(features)
        
        component_probs = self.gmm.predict_proba(features_scaled)
        component_assignments = np.argmax(component_probs, axis=1)
        
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = reducer.fit_transform(features)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=component_assignments, cmap='tab20', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('GMM Component Assignments on Test Data')
        plt.show()
        
        for comp in range(min(5, self.n_components)):
            comp_mask = component_assignments == comp
            if np.sum(comp_mask) > 0:
                comp_labels = labels[comp_mask]
                unique, counts = np.unique(comp_labels, return_counts=True)
                print(f"Component {comp}: {dict(zip(unique, counts))}")

model = GMMSemiSupervisedLearning(n_components=20, n_features=128, n_classes=10)

model.train(labeled_loader, unlabeled_loader, epochs=30)

final_test_acc = model.evaluate(test_loader)
print(f'Final Test Accuracy: {final_test_acc:.2f}%')

model.analyze_gmm_components()

model.visualize_embeddings(test_loader, method='tsne')

model.visualize_gmm_components(test_loader)

model.plot_learning_curves()

print(f"\nGMM-based Semi-Supervised Learning completed with {num_labeled} labeled samples on CIFAR-10")
print(f"Final Test Accuracy: {final_test_acc:.2f}%")
print(f"GMM Components: {model.n_components}")
