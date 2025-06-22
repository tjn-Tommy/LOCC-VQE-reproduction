import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import yaml
import argparse
import logging
from pathlib import Path
import time

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- The Trainer Class ---
class Trainer:
    """A class to encapsulate the training and evaluation loop."""

    def __init__(self, config):
        """
        Initializes the Trainer object.
        Args:
            config (dict): A dictionary containing all configuration parameters.
        """
        self.config = config

        # Setup device
        self.device = self._get_device()
        logging.info(f"Using device: {self.device}")

        # Load data
        self.train_loader, self.test_loader = self._get_data_loaders()

        # Build model, criterion, and optimizer
        self.model = self._build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()

        # State variables
        self.best_accuracy = 0.0
        self.checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_device(self):
        if self.config['training']['device'] == 'cuda' and torch.cuda.is_available():
            return torch.device("cuda")
        if self.config['training']['device'] == 'cuda':
            logging.warning("CUDA not available, falling back to CPU.")
        return torch.device("cpu")

    def _get_data_loaders(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        data_path = Path(self.config['data']['path'])
        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True,
                                                     transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True,
                                                    transform=transform_test)

        train_loader = DataLoader(train_dataset, batch_size=self.config['data']['batch_size'], shuffle=True,
                                  num_workers=self.config['data']['num_workers'])
        test_loader = DataLoader(test_dataset, batch_size=self.config['data']['batch_size'], shuffle=False,
                                 num_workers=self.config['data']['num_workers'])

        return train_loader, test_loader

    def _build_model(self):
        # In a larger project, you could have a model factory here
        if self.config['model']['name'] == 'SimpleCNN':
            return SimpleCNN(num_classes=self.config['model']['num_classes'])
        else:
            raise ValueError(f"Unsupported model: {self.config['model']['name']}")

    def _build_optimizer(self):
        opt_name = self.config['training']['optimizer'].lower()
        if opt_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.config['training']['learning_rate'],
                              weight_decay=self.config['training']['weight_decay'])
        elif opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.config['training']['learning_rate'], momentum=0.9,
                             weight_decay=self.config['training']['weight_decay'])
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    def _train_epoch(self, epoch_num):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % self.config['logging']['log_interval'] == 0:
                logging.info(f"Epoch [{epoch_num}/{self.config['training']['num_epochs']}], "
                             f"Step [{batch_idx + 1}/{len(self.train_loader)}], "
                             f"Loss: {loss.item():.4f}")

    def _evaluate(self):
        """Evaluate the model on the test set."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * data.size(0)

                _, predicted = torch.max(output.data, 1)
                total_samples += target.size(0)
                correct_predictions += (predicted == target).sum().item()

        avg_loss = total_loss / total_samples
        accuracy = 100 * correct_predictions / total_samples
        return avg_loss, accuracy

    def _save_checkpoint(self):
        """Saves the model state if it has the best accuracy so far."""
        best_model_path = self.checkpoint_dir / 'best_model.pth'
        torch.save(self.model.state_dict(), best_model_path)
        logging.info(f"New best model saved to {best_model_path} with accuracy: {self.best_accuracy:.2f}%")

    def train(self):
        """The main training loop."""
        logging.info("Starting training...")
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            start_time = time.time()
            self._train_epoch(epoch)

            val_loss, val_accuracy = self._evaluate()
            epoch_time = time.time() - start_time

            logging.info(f"--- Epoch {epoch} Summary ---")
            logging.info(
                f"Time: {epoch_time:.2f}s, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self._save_checkpoint()

        logging.info("Training finished!")
        logging.info(f"Best validation accuracy: {self.best_accuracy:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 with a YAML config.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("Configuration loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {args.config}")
        exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        exit(1)

    # Create and run the trainer
    trainer = Trainer(config)
    trainer.train()