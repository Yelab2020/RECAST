import os
import random
import copy
from pathlib import Path
from typing import Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset

class VAE(nn.Module):
    """Variational Autoencoder implementation"""
    
    def __init__(
        self,
        input_dim: int,
        enc_hidden_sizes: List[int],
        latent_dim: int,
        dec_hidden_sizes: Optional[List[int]] = None
    ) -> None:
        """
        Args:
            input_dim: Input feature dimension
            enc_hidden_sizes: List of encoder hidden layer dimensions
            latent_dim: Latent space dimension
            dec_hidden_sizes: List of decoder hidden layer dimensions (defaults to symmetric with encoder)
        """
        super().__init__()
        self.latent_dim = latent_dim
        
        # Default decoder structure is symmetric to encoder
        dec_hidden_sizes = dec_hidden_sizes or enc_hidden_sizes[::-1]
        
        # Encoder structure
        enc_layers = []
        prev_size = input_dim
        for size in enc_hidden_sizes:
            enc_layers.append(nn.Linear(prev_size, size))
            enc_layers.append(nn.ReLU())
            prev_size = size
        enc_layers.append(nn.Linear(prev_size, 2 * latent_dim))  # Output mu and logvar
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder structure
        dec_layers = []
        prev_size = latent_dim
        for size in dec_hidden_sizes:
            dec_layers.append(nn.Linear(prev_size, size))
            dec_layers.append(nn.ReLU())
            prev_size = size
        dec_layers.append(nn.Linear(prev_size, input_dim))  # Reconstruct input
        self.decoder = nn.Sequential(*dec_layers)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass"""
        # Encode
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


def create_dataloader(
    data: pd.DataFrame,
    batch_size: int,
    shuffle: bool = True,
    device: Union[str, torch.device] = 'cpu',
    seed: Optional[int] = None
) -> DataLoader:
    """Create dataloader with random seed control"""
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    
    dataset = TensorDataset(torch.tensor(data.values, dtype=torch.float32).to(device))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g if seed is not None else None,
        pin_memory=isinstance(device, str) and device == 'cpu'
    )

def vae_loss(
    recon_x: Tensor,
    x: Tensor,
    mu: Tensor,
    logvar: Tensor,
    kl_weight: float = 1.0
) -> Tensor:
    """Calculate VAE loss function"""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_weight * kl_div


def train_vae(
    data: pd.DataFrame,
    device: Union[str, torch.device] = 'cuda:0',
    latent_dim: int = 128,
    enc_hidden_sizes: List[int] = [2048, 1024, 512],
    batch_size: int = 512,
    epochs: int = 5000,
    lr: float = 1e-4,
    weight_decay: float = 5e-4,
    patience: int = 10000,
    seed: Optional[int] = None
) -> VAE:
    """Train VAE model"""
    
    # Initialize model and optimizer
    model = VAE(
        input_dim=data.shape[1],
        enc_hidden_sizes=enc_hidden_sizes,
        latent_dim=latent_dim
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataloader = create_dataloader(data, batch_size, device=device, seed=seed)  # Fix data order
    
    best_loss = float('inf')
    best_model = None
    early_stop_counter = 0
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch in dataloader:
            x = batch[0].to(device)
            
            # Forward pass
            recon_x, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Early stopping
        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = copy.deepcopy(model)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return best_model


def generate_samples(
    model: VAE,
    data: pd.DataFrame,
    batch_size: int = 512,
    device: Union[str, torch.device] = 'cuda:0',
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate samples (with seed parameter)"""

    
    model.eval()
    dataloader = create_dataloader(data, batch_size, shuffle=False, device=device, seed=seed)
    generated = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            recon_x, _, _ = model(x)
            generated.append(recon_x.cpu().numpy())
    
    return pd.DataFrame(
        data=np.concatenate(generated, axis=0),
        columns=data.columns,
        index=data.index
    )


def train_and_generate(data, seed=114514, **train_kwargs):
    """End-to-end training and data generation"""
    
    # Extract training parameters
    train_params = {k: v for k, v in train_kwargs.items() if k in [
        'device', 'latent_dim', 'enc_hidden_sizes', 'batch_size', 
        'epochs', 'lr', 'weight_decay', 'patience'
    ]}
    
    # Train model
    model = train_vae(data, seed=seed, **train_params)
        
    # Extract generation parameters
    generate_params = {k: v for k, v in train_kwargs.items() if k in [
        'batch_size', 'device'
    ]}    
    
    # Generate samples
    generated_data = generate_samples(model, data, seed=seed, **generate_params)
    
    return generated_data