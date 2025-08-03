from NNfunction import *
import random

# ==== Neural Network Class ====
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, momentum=0.9):
        self.momentum = momentum  # ค่าความเฉื่อยของ momentum

        # กำหนด weight และ bias ระหว่างชั้น input -> hidden
        self.w_ih = [[random_weight() for _ in range(input_size)] for _ in range(hidden_size)]
        self.b_h = [random_weight() for _ in range(hidden_size)]

        # กำหนด weight และ bias ระหว่างชั้น hidden -> output
        self.w_ho = [[random_weight() for _ in range(hidden_size)] for _ in range(output_size)]
        self.b_o = [random_weight() for _ in range(output_size)]

        # ค่า momentum สำหรับการอัปเดต weight/bias
        self.v_w_ih = [[0 for _ in range(input_size)] for _ in range(hidden_size)]
        self.v_b_h = [0 for _ in range(hidden_size)]
        self.v_w_ho = [[0 for _ in range(hidden_size)] for _ in range(output_size)]
        self.v_b_o = [0 for _ in range(output_size)]

    def forward(self, x):  # ฟังก์ชัน feedforward
        # คำนวณ input ของ hidden layer
        self.h_input = [dot(w, x) + b for w, b in zip(self.w_ih, self.b_h)]

        # คำนวณ output ของ hidden layer
        self.h_output = [sigmoid(h) for h in self.h_input]

        # คำนวณ input ของ output layer
        self.o_input = [dot(w, self.h_output) + b for w, b in zip(self.w_ho, self.b_o)]

        # คำนวณ output สุดท้าย
        self.o_output = [sigmoid(o) for o in self.o_input]

        return self.o_output  # คืนค่าผลลัพธ์

    def backward(self, x, y, lr):  # ฟังก์ชัน backpropagation
        # คำนวณ error ที่ output
        output_errors = [(o - y[i]) for i, o in enumerate(self.o_output)]

        # คำนวณ delta ที่ output layer
        output_deltas = [output_errors[i] * sigmoid_derivative(self.o_input[i]) for i in range(len(self.o_output))]

        # คำนวณ error ที่ hidden layer
        hidden_errors = [
            sum(output_deltas[j] * self.w_ho[j][i] for j in range(len(self.o_output)))
            for i in range(len(self.h_output))
        ]

        # คำนวณ delta ที่ hidden layer
        hidden_deltas = [hidden_errors[i] * sigmoid_derivative(self.h_input[i]) for i in range(len(self.h_output))]

        # === อัปเดต weight และ bias ระหว่าง hidden -> output พร้อม momentum ===
        for i in range(len(self.w_ho)):
            for j in range(len(self.w_ho[i])):
                grad = output_deltas[i] * self.h_output[j]
                self.v_w_ho[i][j] = self.momentum * self.v_w_ho[i][j] - lr * grad
                self.w_ho[i][j] += self.v_w_ho[i][j]

            self.v_b_o[i] = self.momentum * self.v_b_o[i] - lr * output_deltas[i]
            self.b_o[i] += self.v_b_o[i]

        # === อัปเดต weight และ bias ระหว่าง input -> hidden พร้อม momentum ===
        for i in range(len(self.w_ih)):
            for j in range(len(self.w_ih[i])):
                grad = hidden_deltas[i] * x[j]
                self.v_w_ih[i][j] = self.momentum * self.v_w_ih[i][j] - lr * grad
                self.w_ih[i][j] += self.v_w_ih[i][j]

            self.v_b_h[i] = self.momentum * self.v_b_h[i] - lr * hidden_deltas[i]
            self.b_h[i] += self.v_b_h[i]

    def train(self, train_data, epochs=1000, lr=0.1, verbose=False):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in train_data:
                output = self.forward(x)  # feedforward
                loss = sum((output[i] - y[i]) ** 2 for i in range(len(y)))  # MSE loss
                total_loss += loss
                self.backward(x, y, lr)  # backpropagation
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    def evaluate(self, val_data):  # ฟังก์ชันประเมินผล
        total_loss = 0
        for x, y in val_data:
            output = self.forward(x)
            loss = sum((output[i] - y[i]) ** 2 for i in range(len(y)))
            total_loss += loss
        return total_loss / len(val_data)


# ==== Load + Normalize Data (Pure Python, no numpy) ====
def load_and_shuffle_data(filename, delimiter="\t"):
    with open(filename, "r") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        if line.strip() == "":
            continue
        row = [float(val) for val in line.strip().split(delimiter)]  # แปลงแต่ละค่าจาก str เป็น float
        data.append(row)

    random.shuffle(data)  # สุ่มลำดับข้อมูล
    return data

def normalize_columns(data, num_input_features):
    X = [row[:num_input_features] for row in data]  # ตัดเฉพาะ input
    y = [[row[num_input_features]] for row in data]  # ตัดเฉพาะ output

    # หาค่า min/max ของแต่ละ column ใน input
    X_min = [min(col) for col in zip(*X)]
    X_max = [max(col) for col in zip(*X)]

    # Normalize input ให้อยู่ในช่วง 0-1
    X_norm = []
    for row in X:
        norm_row = []
        for i, val in enumerate(row):
            min_val = X_min[i]
            max_val = X_max[i]
            norm_val = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.0
            norm_row.append(norm_val)
        X_norm.append(norm_row)

    # Normalize output
    y_vals = [val[0] for val in y]
    y_min = min(y_vals)
    y_max = max(y_vals)
    y_norm = [[(val - y_min) / (y_max - y_min) if y_max != y_min else 0.0] for val in y_vals]

    # รวม input และ output กลับเป็น list เดียว
    normalized_data = [[*X_norm[i], *y_norm[i]] for i in range(len(X_norm))]
    return normalized_data


# ==== K-Fold Cross Validation ====
def k_fold_cross_validation(data, k=10, **nn_params):
    random.shuffle(data)  # สลับข้อมูลอีกครั้งก่อนแบ่ง fold
    fold_size = len(data) // k
    fold_results = []

    for fold in range(k):
        start = fold * fold_size
        end = start + fold_size

        # แบ่ง validation set
        val_data_raw = data[start:end]

        # เอาที่เหลือเป็น training set
        train_data_raw = data[:start] + data[end:]

        # แยก input/output
        train_data = [(row[:-1], [row[-1]]) for row in train_data_raw]
        val_data = [(row[:-1], [row[-1]]) for row in val_data_raw]

        model = NeuralNetwork(**nn_params)  # สร้างโมเดลใหม่ในแต่ละ fold
        model.train(train_data, epochs=1000, lr=0.5)  # เทรนโมเดล

        val_loss = model.evaluate(val_data)  # ประเมินผลบน validation set
        fold_results.append(val_loss)
        print(f"Fold {fold+1}/{k}: Validation Loss = {val_loss:.4f}")

    avg_loss = sum(fold_results) / k  # หาค่าเฉลี่ยของ loss ทั้งหมด
    print(f"\nAverage Validation Loss over {k} folds: {avg_loss:.4f}")
    return avg_loss


# ==== MAIN ==== #
if __name__ == "__main__":
    FILENAME = "Dataset.txt"
    INPUT_SIZE = 8  # จำนวนฟีเจอร์ใน dataset

    raw_data = load_and_shuffle_data(FILENAME)  # โหลดข้อมูลจากไฟล์
    normalized_data = normalize_columns(raw_data, num_input_features=INPUT_SIZE)  # ทำ normalization

    # เรียกใช้งาน K-Fold cross validation
    k_fold_cross_validation(
        normalized_data,
        k=10,  # 10-fold
        input_size=INPUT_SIZE,
        hidden_size=6,  # จำนวน neuron ใน hidden layer
        output_size=1,
        momentum=0.9
    )
