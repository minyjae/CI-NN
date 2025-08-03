import numpy as np
from NNfunction import *

# ===== 1. Load & Shuffle Data =====
data = np.loadtxt("Dataset.txt", delimiter="\t", unpack=False)
np.random.shuffle(data)

# ===== 2. Normalize input (เฉพาะ 8 features) =====
X_all = data[:, :-1]
y_all = data[:, -1:]

X_min = X_all.min(axis=0)
X_max = X_all.max(axis=0)
X_normalized = (X_all - X_min) / (X_max - X_min)

Y_min = y_all.min(axis=0)
Y_max = y_all.max(axis=0)
y_normalized = (y_all - Y_min) / (Y_max - Y_min)

# ===== 3. Combine normalized X and y =====
data_normalized = np.array([np.append(x_row, y_row[0]) for x_row, y_row in zip(X_normalized, y_normalized)])

# ===== 4. Hyperparameters =====
input_size = X_normalized.shape[1]
hidden_size = 10
output_size = 1

learning_rate = 0.01
momentum = 0.9
epochs = 100

# ===== 6. 10-Fold Cross Validation =====
k = 10
n = len(data_normalized)
fold_size = n // k

for fold in range(k):
    print(f"\n===== Fold {fold + 1} =====")
    
    start = fold * fold_size
    end = (fold + 1) * fold_size if fold < k - 1 else n

    val_set = data_normalized[start:end]
    
    # ไม่ใช้ np.concatenate —> ใช้ vstack แบบ manual
    if start == 0:
        train_set = data_normalized[end:]
    elif end == n:
        train_set = data_normalized[:start]
    else:
        train_set = np.vstack([data_normalized[:start], data_normalized[end:]])

    X_train = train_set[:, :-1]
    y_train = train_set[:, -1:].copy()
    X_val = val_set[:, :-1]
    y_val = val_set[:, -1:].copy()

    # === Initialize weights ===
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    # === Initialize momentum terms ===
    vW1 = np.zeros(W1.shape)
    vb1 = np.zeros(b1.shape)
    vW2 = np.zeros(W2.shape)
    vb2 = np.zeros(b2.shape)

    # === Training loop ===
    for epoch in range(epochs):
        z1 = np.dot(X_train, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        y_pred = sigmoid(z2)

        error = y_pred - y_train
        loss = np.mean(error ** 2)

        gradient_output = error * sigmoid_derivative(y_pred)
        gradient_hidden = np.dot(gradient_output, W2.T) * sigmoid_derivative(a1)

        dW2 = np.dot(a1.T, gradient_output)
        db2 = np.sum(gradient_output, axis=0, keepdims=True)
        vW2 = momentum * vW2 - learning_rate * dW2
        vb2 = momentum * vb2 - learning_rate * db2
        W2 += vW2
        b2 += vb2

        dW1 = np.dot(X_train.T, gradient_hidden)
        db1 = np.sum(gradient_hidden, axis=0, keepdims=True)
        vW1 = momentum * vW1 - learning_rate * dW1
        vb1 = momentum * vb1 - learning_rate * db1
        W1 += vW1
        b1 += vb1

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}: Training Loss = {loss:.6f}")

    # === Validate ===
    z1_val = np.dot(X_val, W1) + b1
    a1_val = sigmoid(z1_val)
    z2_val = np.dot(a1_val, W2) + b2
    y_val_pred = sigmoid(z2_val)
    val_loss = np.mean((y_val_pred - y_val) ** 2)

    print(f"Final Training Loss: {loss:.6f}")
    print(f"Validation Loss: {val_loss:.6f}")
