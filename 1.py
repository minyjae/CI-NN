import numpy as np
from NNfunction import *

# ===== 1. Load & Shuffle Data =====
data = np.loadtxt("Dataset.txt", delimiter="\t", unpack=False) # อ่านไฟล์ dataset.txt 
# np.random.seed(42) # set seed ในการ random
np.random.shuffle(data) # สลับลำดับของ dataset

# ===== 2. Normalize input (เฉพาะ 8 features) =====
X_all = data[:, :-1] # ดึงให้ input = 8 ในแต่ละแถว
y_all = data[:, -1:] # ดึง y true ตัวสุดท้ายออกมา ในแต่ละแถว

X_min = X_all.min(axis=0) # หาค่า min ของ input
X_max = X_all.max(axis=0) # หาค่า max ของ input
X_normalized = (X_all - X_min) / (X_max - X_min) # ทำการ normalized ค่าให้มีค่า 0 ถึง 1

Y_min = y_all.min(axis=0) # หาค่า min ของ input
Y_max = y_all.max(axis=0) # หาค่า max ของ input
y_normalized = (y_all - Y_min) / (Y_max - Y_min) # ทำการ normalized ค่าให้มีค่า 0 ถึง 1

# ===== 3. Combine normalized X and y =====
data_normalized = np.hstack((X_normalized, y_normalized)) # นำ dataset ที่ normalized มารวมกัน

# ===== 4. Hyperparameters =====
input_size = X_normalized.shape[1] # input_size มีขนาดเท่ากับ จำนวน X_normalized แถวนึง = 8
hidden_size = 10 # จำนวน node ใน hidden layer
output_size = 1 # จำนวน output 

learning_rate = 0.01 # learning rate
momentum = 0.9 # momentum
epochs = 100 # จำนวน epochs ที่จะ train

# ===== 6. 10-Fold Cross Validation =====
k = 10 # จำนวน fold ที่จะทำ
n = len(data_normalized) # จำนวนของ dataset
fold_size = n // k # fold size

for fold in range(k):
    print(f"\n===== Fold {fold + 1} =====")
    
    # === Split train/val sets ===
    start = fold * fold_size # จุด start dataset ของแต่ละ fold ว่าจะเริ่มจุดไหน พอ fold มากขึ้นก็จะเริ่มตามที่แบ่ง fold ไว้
    end = (fold + 1) * fold_size if fold < k - 1 else n # จุด end dataset

    val_set = data_normalized[start:end] # ยกตัวอย่าง fold แรก ก็จะเอา dataset ที่ 1-31
    train_set = np.concatenate((data_normalized[:start], data_normalized[end:]), axis=0) # training set จะเป็นตั้งแต่ 32 เป็นต้นไป

    X_train, y_train = train_set[:, :-1], train_set[:, -1:].copy() # แยกระหว่าง ค่า input และ y true ของ training set
    X_val, y_val = val_set[:, :-1], val_set[:, -1:].copy() # แยกระหว่าง ค่า input และ y true ของ training set

    # === Initialize weights ===
    # np.random.seed(0) # set seed ในการ random
    W1 = np.random.randn(input_size, hidden_size) # สุ่ม weight input -> hidden
    b1 = np.zeros((1, hidden_size)) # สุ่ม bias hidden
    W2 = np.random.randn(hidden_size, output_size) # สุ่ม weight hidden -> output
    b2 = np.zeros((1, output_size)) # สุ่ม bias output

    # === Initialize momentum terms ===
    vW1 = np.zeros_like(W1) # เก็บ weight เดิม ของ input -> hidden
    vb1 = np.zeros_like(b1) # เก็บ bias เติม ของ hidden 
    vW2 = np.zeros_like(W2) # เก็บ weight เดิม ของ hidden -> output
    vb2 = np.zeros_like(b2) # เก็บ bias เดิม ของ output

    # === Training loop ===
    for epoch in range(epochs):
        # ---- Forward ----
        z1 = np.dot(X_train, W1) + b1 # นำ training set กับ weight มาคูณกัน แล้วบวก bias(hidden)
        a1 = sigmoid(z1) # ผ่าน sigmoid 
        z2 = np.dot(a1, W2) + b2  # นำ training set กับ weight มาคูณกัน แล้วบวก bias (output)
        y_pred = sigmoid(z2) # ผ่าน sigmoid

        # ---- Loss ----
        error = y_pred - y_train # error = y ที่ได้ออกมาจากการ train ในแต่ละรอบ ลบกับ y_true จาก dataset
        loss = np.mean(error ** 2) # หา mean square error ในการ

        # ---- Gradient output ----
        gradient_output = error * sigmoid_derivative(y_pred)  # (y_pred - y) * y_pred * (1 - y_pred)

        # ---- Gradient hidden ----
        gradient_hidden = np.dot(gradient_output, W2.T) * sigmoid_derivative(a1) 

        # ---- Weight updates with Momentum ----
        # Output weights and bias
        dW2 = np.dot(a1.T, gradient_output) # ส่วนของ gradient_output คูณกับ y ที่ออกมาจาก hidden
        db2 = np.sum(gradient_output, axis=0, keepdims=True) # bias update โดยใช้แค่ gradient

        vW2 = momentum * vW2 - learning_rate * dW2 # delta weight (hidden -> output)
        vb2 = momentum * vb2 - learning_rate * db2 # delta bias (output)

        W2 += vW2 # update weight (hidden -> output)
        b2 += vb2 # update bias (output)

        # Hidden weights and bias
        dW1 = np.dot(X_train.T, gradient_hidden) # ส่วนของ gradient_hidden คูณกับ input
        db1 = np.sum(gradient_hidden, axis=0, keepdims=True) # bias update โดยใช้แค่ gradient

        vW1 = momentum * vW1 - learning_rate * dW1 # delta weight (input -> hidden)
        vb1 = momentum * vb1 - learning_rate * db1 # delta bias (hidden)

        W1 += vW1 # update weight (input -> hidden)
        b1 += vb1 # update bias (hidden)

        # ---- Print some logs ----
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}: Training Loss = {loss:.6f}")

    # === Validate ===
    # forward pass ปกติ เพื่อหา mean square loss (ผมรวมของ error เมื่อผ่าน nn ทั้งหมดแล้ว) ของ validate set
    z1_val = np.dot(X_val, W1) + b1
    a1_val = sigmoid(z1_val)
    z2_val = np.dot(a1_val, W2) + b2
    y_val_pred = sigmoid(z2_val)
    val_loss = np.mean((y_val_pred - y_val) ** 2)

    print(f"Final Training Loss: {loss:.6f}")
    print(f"Validation Loss: {val_loss:.6f}")
