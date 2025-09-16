import gzip
import os
from pprint import pprint
from urllib.request import urlretrieve

import numpy as np
from autograd import CrossEntropyLoss, ReLU, Sum, Tensor


def load_mnist(path=None):
    """Download MNIST and load it into NumPy arrays."""
    # url_base = "http://yann.lecun.com/exdb/mnist/"
    url_base = "https://ossci-datasets.s3.amazonaws.com/mnist/"

    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    # Default path to ./data/mnist
    if path is None:
        path = os.path.join(os.path.expanduser("./"), "data", "mnist")
    os.makedirs(path, exist_ok=True)

    # Download missing files
    for name, filename in files.items():
        filepath = os.path.join(path, filename)
        if not os.path.isfile(filepath):
            urlretrieve(url_base + filename, filepath)
            print(f"Downloaded {filename}")

    # Load images
    def _read_images(filename):
        with gzip.open(os.path.join(path, filename), "rb") as f:
            data = f.read()
            # The first 16 bytes are header (magic, num, rows, cols)
            images = np.frombuffer(data, dtype=np.uint8, offset=16)
        images = images.reshape(-1, 28 * 28).astype(np.float32) / 255.0
        return images

    # Load labels
    def _read_labels(filename):
        with gzip.open(os.path.join(path, filename), "rb") as f:
            data = f.read()
            # First 8 bytes are header (magic, num)
            labels = np.frombuffer(data, dtype=np.uint8, offset=8)
        # Convert to one-hot vectors of length 10
        one_hot = np.zeros((labels.size, 10), dtype=np.float32)
        one_hot[np.arange(labels.size), labels] = 1.0
        return one_hot

    # Read all parts
    X_train = _read_images(files["train_images"])
    y_train = _read_labels(files["train_labels"])
    X_test = _read_images(files["test_images"])
    y_test = _read_labels(files["test_labels"])
    return X_train, y_train, X_test, y_test


# Initialize network parameters
class MLP3:
    def __init__(self, in_dim=784, h1=128, h2=64, out_dim=10, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = Tensor(rng.normal(0, 0.01, (in_dim, h1)), requires_grad=True)
        self.b1 = Tensor(np.zeros(h1, dtype=np.float32), requires_grad=True)
        self.W2 = Tensor(rng.normal(0, 0.01, (h1, h2)), requires_grad=True)
        self.b2 = Tensor(np.zeros(h2, dtype=np.float32), requires_grad=True)
        self.W3 = Tensor(rng.normal(0, 0.01, (h2, out_dim)), requires_grad=True)
        self.b3 = Tensor(np.zeros(out_dim, dtype=np.float32), requires_grad=True)

    @property
    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def __call__(self, x: Tensor) -> Tensor:
        h1 = ReLU.apply(x @ self.W1 + self.b1)
        h2 = ReLU.apply(h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        return logits


def run_mnist(
    model,
    learning_rate,
    batch_size,
    num_epochs,
    l2_weight,
):
    X_train, y_train, X_test, y_test = load_mnist()
    for epoch in range(num_epochs):
        # Shuffle training data
        perm = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[perm], y_train[perm]
        total_loss = 0.0

        for i in range(0, X_train.shape[0], batch_size):
            # Mini-batch slice
            X_batch = Tensor(X_train[i : i + batch_size])  # input batch
            y_batch = Tensor(y_train[i : i + batch_size])  # target batch

            loss_l2 = Tensor(l2_weight) * (
                Sum.apply(model.W1**2) + Sum.apply(model.W2**2) + Sum.apply(model.W3**2)
            )

            # Forward pass:
            logits = model(X_batch)
            # probs = Softmax.apply(logits)
            # logp = Log.apply(probs)
            # loss = NLLLoss.apply(logp, y_batch) + loss_l2
            loss = (
                CrossEntropyLoss.apply(logits, y_batch) + loss_l2
            )  # Compute cross-entropy loss over the batch
            total_loss += float(loss.data)

            # Backward pass:
            loss.backward()  # compute gradients for all weights/biases

            # Update parameters with SGD:
            for param in model.params:
                # simple gradient descent step
                param.data -= learning_rate * param.grad
                # reset gradient to zero for next batch
                param.grad = None

        # quick test accuracy
        Xt = Tensor(X_test, requires_grad=False)
        logits = model(Xt)
        preds = np.argmax(logits.data, axis=1)
        true = np.argmax(y_test, axis=1)
        acc = (preds == true).mean()

    return acc


def test_l2_weights(l2_weights, runs):
    results = {}
    for l2_weight in l2_weights:
        accs = []
        for _ in range(runs):
            acc = run_mnist(
                model=MLP3(),
                learning_rate=0.1,
                batch_size=100,
                num_epochs=10,
                l2_weight=l2_weight,
            )
            accs.append(acc)
            print(acc)

        results[l2_weight] = sum(accs) / len(accs)
        pprint(results)

    pprint(results)

    return results


if __name__ == "__main__":
    test_l2_weights([0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 50)
