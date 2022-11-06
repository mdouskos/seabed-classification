from typing import List
import torch.nn
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import logging

from torch.utils.data import TensorDataset


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


class NNSeabedClasifier(BaseEstimator):
    def __init__(self, hidden_sizes: List, epochs=10, batch_size=2048, gpu=False):
        assert len(hidden_sizes) > 0, "At least one hidden layer is needed"
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.epochs = epochs
        self.gpu = gpu

    def __init_seq(self, D_in, num_classes, activation=torch.nn.ReLU, dropout=True):
        # Build sequential model from list of layers
        layerlist = [torch.nn.Linear(D_in, self.hidden_sizes[0])]
        for i in range(len(self.hidden_sizes) - 1):
            layerlist.append(activation())
            layerlist.append(
                torch.nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1])
            )
        layerlist.append(activation())
        if dropout:
            layerlist.append(torch.nn.Dropout())
        layerlist.append(torch.nn.Linear(self.hidden_sizes[-1], num_classes))
        layerlist.append(torch.nn.Softmax(dim=1))
        self.seq = torch.nn.Sequential(*layerlist)

        if self.gpu:
            self.seq = self.seq.cuda()

        logger.debug(str(self.seq))

    def fit(self, X, y):
        check_X_y(X, y)

        self.input_dim = X.shape[1]
        self.num_classes = len(np.unique(y))
        self.__init_seq(self.input_dim, self.num_classes)

        Xtnsr = torch.tensor(X)
        ytnsr = torch.tensor(y)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.seq.parameters(), lr=0.001)

        train_loader = torch.utils.data.DataLoader(
            TensorDataset(Xtnsr, ytnsr), batch_size=self.batch_size
        )
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for data in train_loader:
                samples = data[0]
                labels = data[1]
                samples = samples.type(torch.FloatTensor)
                labels = labels.type(torch.LongTensor)
                if self.gpu:
                    samples = samples.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                outputs = self.seq(samples)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += outputs.shape[0] * loss.item()
            training_set_accuracy = np.sum(
                np.argmax(self.__compute_logits(X), 1) == y
            ) / len(y)
            logger.debug(f"Training set accuracy: {training_set_accuracy:.3f}")
            logger.info(
                "Epoch: {}/{}, loss: {}".format(
                    epoch + 1, self.epochs, epoch_loss / len(ytnsr)
                )
            )
        self.fitted_ = True

    def __compute_logits(self, X):
        logits = np.empty((0, self.num_classes), dtype=np.uint8)

        data_loader = torch.utils.data.DataLoader(
            TensorDataset(torch.tensor(X)), batch_size=self.batch_size
        )
        with torch.no_grad():
            for data in data_loader:
                samples = data[0]
                samples = samples.type(torch.FloatTensor)
                if self.gpu:
                    samples = samples.cuda()
                batch_logits = self.seq(samples)
                logits = np.vstack((logits, batch_logits.detach().cpu().numpy()))
        return logits

    def predict_proba(self, X):
        check_is_fitted(self)
        check_array(X)

        return self.__compute_logits(X)

    def predict(self, X):
        check_is_fitted(self)
        check_array(X)

        return np.argmax(self.__compute_logits(X), 1)


seabed_classifier_type = {
    "rf": RandomForestClassifier,
    "svm": LinearSVC,
    "nn": NNSeabedClasifier,
}

seabed_classifier_name = {
    "rf": "Random Forest",
    "svm": "Support Vector Machine",
    "nn": "Neural Network",
}
