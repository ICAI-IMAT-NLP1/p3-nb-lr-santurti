import torch

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class LogisticRegression:
    def __init__(self, random_state: int):
        self._weights: torch.Tensor = None
        self.random_state: int = random_state

    def fit(self, features: torch.Tensor, labels: torch.Tensor, learning_rate: float, epochs: int):
        self.weights = self.initialize_parameters(features.size(1) + 1, self.random_state)  # Include bias in weights
        
        for _ in range(epochs):
            features_with_bias = torch.cat((features, torch.ones((features.size(0), 1))), dim=1)  # Add bias as feature
            z = torch.matmul(features_with_bias, self.weights)
            pred = self.sigmoid(z)
            loss = self.binary_cross_entropy_loss(predictions=pred, targets=labels)
            
            gradient_weights = torch.matmul(features_with_bias.T, (pred - labels)) / labels.size(0)
            
            self.weights -= learning_rate * gradient_weights

    def predict(self, features: torch.Tensor, cutoff: float = 0.5) -> torch.Tensor:
        probabilities = self.predict_proba(features)
        return (probabilities >= cutoff).float()

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise ValueError("Model not trained. Call the 'fit' method first.")
        features_with_bias = torch.cat((features, torch.ones((features.size(0), 1))), dim=1)  # Add bias as feature
        return self.sigmoid(torch.matmul(features_with_bias, self.weights))

    def initialize_parameters(self, dim: int, random_state: int) -> torch.Tensor:
        torch.manual_seed(random_state)
        return torch.randn(dim)

    @staticmethod
    def sigmoid(z: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-z))

    @staticmethod
    def binary_cross_entropy_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-10
        predictions = torch.clamp(predictions, min=epsilon, max=1 - epsilon)
        return - (targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions)).mean()

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value
