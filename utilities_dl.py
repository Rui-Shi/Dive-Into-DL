from random import random
import time
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# --- 1. Class Extension Utilities ---

def add_to_class(Class): #@save
    """Register functions as methods in created class.
    
    Commonly used in notebooks to split large class definitions 
    across multiple cells for better readability.
    """
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

# --- 2. Base Configuration & Visualization ---

class HyperParameters: #@save
    """The base class of hyperparameters.
    
    Intended to be implemented to automatically store configuration 
    arguments in 'self' to reduce boilerplate.
    """
    def save_hyperparameters(self, ignore=[]):
        raise NotImplementedError

class ProgressBoard(d2l.HyperParameters): #@save
    """The board that plots data points in animation.
    
    Used for real-time visualization of training metrics.
    """
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        """Placeholder for plotting logic."""
        raise NotImplementedError

# --- 3. The Core Deep Learning Trinity ---

class Module(nn.Module, d2l.HyperParameters): #@save
    """The base class of models.
    
    Combines PyTorch's nn.Module with automatic hyperparameter 
    saving and built-in plotting capabilities.
    """
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network net is not defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation during training/validation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches
            n = self.trainer.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / self.plot_valid_per_epoch
        
        self.board.draw(x, value.to(d2l.cpu()).detach().numpy(),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError

class DataModule(d2l.HyperParameters): #@save
    """The base class of data pipelines.
    
    Standardizes how training and validation data loaders are accessed.
    """
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

class Trainer(d2l.HyperParameters): #@save
    """The base class for training models with data.
    
    Manages the orchestration of the training loop across epochs.
    """
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        """Placeholder for inner loop batch processing."""
        raise NotImplementedError


# Inherits from the DataModule class to standardize how data is fed to the Trainer
class SyntheticRegressionData(d2l.DataModule): #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        super().__init__()
        # Automatically saves w, b, noise, num_train, etc., to self
        self.save_hyperparameters()
        
        # Total number of data points needed (train + validation)
        n = num_train + num_val
        
        # 1. Generate the Features (X)
        # Creates a matrix of size (n, length of w) filled with random numbers 
        # from a standard normal distribution (mean=0, variance=1)
        self.X = torch.randn(n, len(w))
        
        # 2. Generate the Noise
        # Creates a column vector of random noise, scaled by the 'noise' parameter
        noise_tensor = torch.randn(n, 1) * noise
        
        # 3. Generate the Labels (y)
        # Performs matrix multiplication: X * w + b + noise
        # w.reshape((-1, 1)) ensures w is treated as a column vector
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise_tensor


# Injects this method into the SyntheticRegressionData class defined earlier
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    # 1. Determine which subset of data to use
    if train:
        # Create a list of indices for the training data (e.g., 0 to 999)
        indices = list(range(0, self.num_train))
        # Randomly shuffle the training indices. 
        # This is crucial for Stochastic Gradient Descent (SGD) to work properly!
        random.shuffle(indices)
    else:
        # Create a list of indices for the validation data (e.g., 1000 to 1999)
        # We don't need to shuffle validation data because we are just evaluating it.
        indices = list(range(self.num_train, self.num_train + self.num_val))
        
    # 2. Group the indices into batches
    # Iterate from 0 to the end of the data, stepping by 'batch_size' (e.g., 32)
    for i in range(0, len(indices), self.batch_size):
        # Grab the next 32 indices and convert them to a PyTorch tensor
        batch_indices = torch.tensor(indices[i: i+self.batch_size])
        
        # 3. Yield the actual data points
        # Return the corresponding slice of Features (X) and Labels (y)
        yield self.X[batch_indices], self.y[batch_indices]


@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    # 1. Slicing: Subsets the data (X and y) based on the provided indices
    tensors = tuple(a[indices] for a in tensors)
    
    # 2. Wrapping: Pairs the corresponding X and y rows together
    dataset = torch.utils.data.TensorDataset(*tensors)
    
    # 3. Loading: Creates the PyTorch DataLoader 
    return torch.utils.data.DataLoader(dataset, self.batch_size,
                                       shuffle=train)


@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    # 1. Slicing: Subsets the data (X and y) based on the provided indices
    tensors = tuple(a[indices] for a in tensors)
    
    # 2. Wrapping: Pairs the corresponding X and y rows together
    dataset = torch.utils.data.TensorDataset(*tensors)
    
    # 3. Loading: Creates the PyTorch DataLoader 
    return torch.utils.data.DataLoader(dataset, self.batch_size,
                                       shuffle=train)