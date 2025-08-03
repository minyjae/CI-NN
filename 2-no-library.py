from NNfunction import *
import random                     # ใช้สำหรับสุ่มค่า เช่น น้ำหนัก และการสับข้อมูล
import matplotlib.pyplot as plt   # ใช้สำหรับ plot กราฟ

# ==== Neural Network Class ====
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, momentum=0.9):
        # กำหนดขนาดของ layer และค่า momentum
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.momentum = momentum

        # สุ่มค่าเริ่มต้นของน้ำหนักและ bias สำหรับ input -> hidden
        self.w_ih = [[random_weight() for _ in range(input_size)] for _ in range(hidden_size)]
        self.b_h = [random_weight() for _ in range(hidden_size)]

        # สุ่มค่าเริ่มต้นของน้ำหนักและ bias สำหรับ hidden -> output
        self.w_ho = [[random_weight() for _ in range(hidden_size)] for _ in range(output_size)]
        self.b_o = [random_weight() for _ in range(output_size)]

        # สร้างตัวเก็บ momentum สำหรับการอัปเดตน้ำหนัก
        self.v_w_ih = [[0 for _ in range(input_size)] for _ in range(hidden_size)]
        self.v_b_h = [0 for _ in range(hidden_size)]
        self.v_w_ho = [[0 for _ in range(hidden_size)] for _ in range(output_size)]
        self.v_b_o = [0 for _ in range(output_size)]

        self.loss_history = []  # เก็บค่า loss ในแต่ละ epoch

    def forward(self, x):  # การ feedforward
        self.h_input = [dot(w, x) + b for w, b in zip(self.w_ih, self.b_h)]     # คำนวณค่า input ของ hidden layer
        self.h_output = [sigmoid(h) for h in self.h_input]                     # นำค่าไปผ่าน sigmoid
        self.o_input = [dot(w, self.h_output) + b for w, b in zip(self.w_ho, self.b_o)]  # คำนวณ input ของ output layer
        self.o_output = [sigmoid(o) for o in self.o_input]                     # นำค่าไปผ่าน sigmoid
        return self.o_output  # คืนค่า output

    def backward(self, x, y, lr):  # การทำ backpropagation
        output_errors = [(o - y[i]) for i, o in enumerate(self.o_output)]   # คำนวณ error ที่ output
        output_deltas = [output_errors[i] * sigmoid_derivative(self.o_input[i]) for i in range(self.output_size)]

        # คำนวณ error และ delta ของ hidden layer
        hidden_errors = [
            sum(output_deltas[j] * self.w_ho[j][i] for j in range(self.output_size))
            for i in range(self.hidden_size)
        ]
        hidden_deltas = [hidden_errors[i] * sigmoid_derivative(self.h_input[i]) for i in range(self.hidden_size)]

        # อัปเดตน้ำหนัก hidden -> output
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                grad = output_deltas[i] * self.h_output[j]
                self.v_w_ho[i][j] = self.momentum * self.v_w_ho[i][j] - lr * grad
                self.w_ho[i][j] += self.v_w_ho[i][j]

            self.v_b_o[i] = self.momentum * self.v_b_o[i] - lr * output_deltas[i]
            self.b_o[i] += self.v_b_o[i]

        # อัปเดตน้ำหนัก input -> hidden
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                grad = hidden_deltas[i] * x[j]
                self.v_w_ih[i][j] = self.momentum * self.v_w_ih[i][j] - lr * grad
                self.w_ih[i][j] += self.v_w_ih[i][j]

            self.v_b_h[i] = self.momentum * self.v_b_h[i] - lr * hidden_deltas[i]
            self.b_h[i] += self.v_b_h[i]

    def train(self, train_data, epochs, lr, converge_threshold):  # ฟังก์ชันสำหรับฝึกโมเดล
        self.loss_history = []  # เคลียร์ history ก่อนฝึก
        for epoch in range(epochs):
            total_loss = 0  # รวม loss ทั้งหมดใน epoch นี้
            for x, y in train_data:
                output = self.forward(x)  # ทำ forward
                loss = sum((output[i] - y[i]) ** 2 for i in range(len(y)))  # คำนวณ loss
                total_loss += loss
                self.backward(x, y, lr)  # ทำ backpropagation

            avg_loss = total_loss / len(train_data)  # คำนวณค่าเฉลี่ย loss
            self.loss_history.append(avg_loss)  # บันทึกลง history

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}: Training Loss = {avg_loss:.6f}")

            if avg_loss < converge_threshold:  # ตรวจสอบว่า loss ต่ำพอหรือยัง
                return epoch + 1  # หยุดการเรียนรู้
        return epochs  # เรียนจนครบ epoch

    def evaluate(self, val_data):  # ใช้ประเมินโมเดลบน validation set
        total_loss = 0
        correct = 0
        TP = TN = FP = FN = 0

        for x, y in val_data:
            output = self.forward(x)
            loss = sum((output[i] - y[i]) ** 2 for i in range(len(y)))
            total_loss += loss

            pred_class = argmax(output)
            true_class = argmax(y)

            if pred_class == true_class:
                correct += 1
                if pred_class == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if pred_class == 1:
                    FP += 1
                else:
                    FN += 1

        avg_loss = total_loss / len(val_data)
        accuracy = correct / len(val_data)
        return avg_loss, accuracy, TP, FP, FN, TN


# ==== Load .pat File ====
def load_cross_pat(filename):  # ฟังก์ชันโหลดข้อมูลจากไฟล์ .pat
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("p"):  # ถ้าพบ pattern ใหม่
            input_line = list(map(float, lines[i+1].strip().split()))  # อ่าน input
            output_line = list(map(int, lines[i+2].strip().split()))   # อ่าน output
            data.append(input_line + output_line)                      # รวมเก็บไว้
            i += 3
        else:
            i += 1
    return data


# ==== Main ====
if __name__ == "__main__":
    FILENAME = "cross.pat"                         # ชื่อไฟล์ข้อมูล
    raw_data = load_cross_pat(FILENAME)            # โหลดข้อมูลจากไฟล์
    random.shuffle(raw_data)                       # สุ่มลำดับข้อมูล

    input_size = 2                                  # จำนวน input nodes
    output_size = 2                                 # จำนวน output nodes
    hidden_size = 15                                # จำนวน hidden nodes
    momentum = 0.9                                  # ค่า momentum
    learning_rate = 0.01                            # ค่า learning rate
    epochs = 1000                                   # จำนวนรอบฝึกสูงสุด
    converge_threshold = 0.01                       # เกณฑ์สำหรับการหยุดเมื่อ loss ต่ำพอ
    k = 10                                          # จำนวน fold สำหรับ cross validation
    fold_size = len(raw_data) // k
    converge_epochs_all = []                        # เก็บ epoch ที่แต่ละ fold converged
    all_loss_histories = []                         # เก็บค่า loss history ของแต่ละ fold

    # Normalize เฉพาะ input เท่านั้น
    inputs = [row[:input_size] for row in raw_data]
    outputs = [row[input_size:] for row in raw_data]
    input_mins = [min(col) for col in zip(*inputs)]
    input_maxs = [max(col) for col in zip(*inputs)]

    normalized_data = []
    for i in range(len(inputs)):
        norm_input = [
            (inputs[i][j] - input_mins[j]) / (input_maxs[j] - input_mins[j])
            if input_maxs[j] != input_mins[j] else 0.0
            for j in range(input_size)
        ]
        normalized_data.append((norm_input, outputs[i]))

    # ทำ k-Fold cross validation
    for fold in range(k):
        print(f"\n===== Fold {fold+1} =====")
        start = fold * fold_size
        end = start + fold_size if fold < k - 1 else len(normalized_data)

        val_data = normalized_data[start:end]
        train_data = normalized_data[:start] + normalized_data[end:]

        model = NeuralNetwork(input_size, hidden_size, output_size, momentum)
        converge_epoch = model.train(train_data, epochs, learning_rate, converge_threshold)

        val_loss, accuracy, TP, FP, FN, TN = model.evaluate(val_data)

        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Converged at Epoch: {converge_epoch}")
        print(f"Confusion Matrix: TP={TP}, FP={FP}, FN={FN}, TN={TN}")

        converge_epochs_all.append(converge_epoch)
        all_loss_histories.append(model.loss_history)

    # สรุปผลลัพธ์จากทุก fold
    print("\n=== Convergence Summary ===")
    for i, epoch in enumerate(converge_epochs_all):
        print(f"Fold {i+1}: Converged at Epoch {epoch}")
    avg_epoch = sum(converge_epochs_all) / len(converge_epochs_all)
    print(f"Average Convergence Epoch: {avg_epoch:.2f}")

    # ==== Plot Loss Convergence ====
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(all_loss_histories):
        plt.plot(history, label=f"Fold {i+1}")
    plt.title("Training Loss per Epoch (Each Fold)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
