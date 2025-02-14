import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        labels_int = labels.to(torch.int64)
        class_counts = torch.bincount(labels_int)
        self.class_priors = class_counts / labels.size(0)

        # Vocabulary size is the number of features
        self.vocab_size = features.size(1)

        # Estimate conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # TODO: Count number of samples for each output class and divide by total of samples

        class_priors: Dict[int, torch.Tensor] = {}
        nlabels: int = len(labels)

        for label in labels:
            label = label.item()
            if label in class_priors:
                class_priors[label] = (class_priors[label] * nlabels + 1) / nlabels
            else:
                class_priors[label] = torch.tensor(1,dtype=torch.float64) / nlabels
        
        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # Estimate conditional probabilities for the words in features and apply smoothing

        class_word_counts: Dict[int, torch.Tensor] = {}

        # Create dict elements (adding Laplace's delta)
        for label in torch.unique(labels):
            class_word_counts[label.item()] = torch.full(size=(features.size(1),), fill_value=delta)
            
        # Adjust dict elements
        for i in range(features.size(0)):
            label = labels[i].item()
            class_word_counts[label] += features[i]

        probab: Dict[int, torch.Tensor] = {}
        for key, tensor in class_word_counts.items():
            total = tensor.sum()
            probab[key] = tensor / total

        return probab


    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # TODO: Calculate posterior based on priors and conditional probabilities of the words
        log_priors = torch.log(self.class_priors)
        log_likelihoods = torch.zeros(len(self.class_priors))

        for class_idx, conditional_probs in self.conditional_probabilities.items():
            class_idx = int(class_idx)
            log_likelihoods[class_idx] = (feature * torch.log(conditional_probs)).sum()
    
        log_posteriors: torch.Tensor = log_priors + log_likelihoods
        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if self.class_priors is None or self.conditional_probabilities is None:
            raise Exception("Model not trained. Please call the train method first.")
        
           # Calculate log posteriors and obtain the class of maximum likelihood
        log_posteriors = self.estimate_class_posteriors(feature)
        pred = torch.argmax(log_posteriors).item()  # Get the class with the highest posterior probability

        return pred


    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if self.class_priors is None or self.conditional_probabilities is None:
            raise Exception("Model not trained. Please call the train method first.")

        log_posteriors = self.estimate_class_posteriors(feature)
        
        # Transform log posteriors to probabilities using softmax
        probs = torch.softmax(log_posteriors, dim=0)

        return probs
