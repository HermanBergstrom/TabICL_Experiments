"""
PyTorch modules for baseline classification heads.
Implements linear probing and 2-layer MLP for multimodal learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


class LinearProbe(nn.Module):
    """
    Linear probing head for classification.
    Single linear layer for direct classification.
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        """
        Args:
            input_dim: Dimension of input features.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPHead(nn.Module):
    """
    2-layer MLP head for classification.
    Linear -> ReLU -> Dropout -> Linear
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of input features.
            num_classes: Number of output classes.
            hidden_dim: Dimension of hidden layer.
            dropout: Dropout probability.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def train_head(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    learning_rate: float = 0.001,
    num_epochs: int = 100,
    batch_size: int = 32,
    early_stopping_patience: int = 10,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple:
    """
    Train a classification head (linear probe or MLP).
    
    Args:
        model: PyTorch module to train.
        X_train: Training features (numpy array).
        y_train: Training targets (numpy array).
        X_val: Validation features (numpy array).
        y_val: Validation targets (numpy array).
        num_classes: Number of output classes.
        learning_rate: Learning rate for optimizer.
        num_epochs: Maximum number of epochs.
        batch_size: Batch size for training.
        early_stopping_patience: Patience for early stopping.
        device: Device to use ("cpu" or "cuda").
        verbose: Whether to print training progress.
    
    Returns:
        Tuple of (model, best_metrics)
    """
    
    # Move model to device
    model = model.to(device)
    
    # Convert numpy arrays to torch tensors
    X_train_tensor = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_train_tensor = torch.from_numpy(y_train.astype(np.int64)).to(device)
    X_val_tensor = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_val_tensor = torch.from_numpy(y_val.astype(np.int64)).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = -np.inf
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor)
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
        
        # Compute metrics
        val_accuracy = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss / len(train_loader):.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, "
                  f"Val Acc: {val_accuracy:.4f}, "
                  f"Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Compute final metrics
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
        val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
    
    metrics = {
        "accuracy": accuracy_score(y_val, val_preds),
        "precision": precision_score(y_val, val_preds, average="weighted", zero_division=0),
        "recall": recall_score(y_val, val_preds, average="weighted", zero_division=0),
        "f1": f1_score(y_val, val_preds, average="weighted", zero_division=0),
    }
    
    # Compute AUROC
    try:
        metrics["auroc"] = roc_auc_score(y_val, val_probs, multi_class="ovr", average="weighted")
    except Exception:
        metrics["auroc"] = None
    
    return model, metrics


def predict_head(
    model: nn.Module,
    X: np.ndarray,
    device: str = "cpu",
) -> tuple:
    """
    Make predictions using a trained head.
    
    Args:
        model: Trained PyTorch module.
        X: Features (numpy array).
        device: Device to use ("cpu" or "cuda").
    
    Returns:
        Tuple of (predictions, probabilities)
    """
    model.eval()
    X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    
    return preds, probs
