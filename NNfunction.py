import random
import math

# ==== Activation Function ====
def sigmoid(x):  # ฟังก์ชัน sigmoid สำหรับทำ activation
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):  # อนุพันธ์ของ sigmoid ใช้ใน backpropagation
    sx = sigmoid(x)
    return sx * (1 - sx)

def dot(a, b):  # ฟังก์ชันคูณเวกเตอร์จุด (dot product)
    return sum([x * y for x, y in zip(a, b)])

def random_weight():  # สุ่มค่าน้ำหนักเริ่มต้นระหว่าง -1 ถึง 1
    return random.uniform(-1, 1)

def argmax(vec):                 # คืนค่าตำแหน่งของค่ามากที่สุดในเวกเตอร์
    return max(range(len(vec)), key=lambda i: vec[i])

# def gradientOut(e,y):
#     g=e*y*(1-y)
#     return (g)


# def gradientHidden(y,sum):
#     g=y*(1-y)*sum
#     return (g)

# def deltaw(m, l, g, x, dw_prev):
#     """
#     คำนวณ delta weight พร้อม momentum
#     m: ค่า momentum (เช่น 0.9)
#     l: learning rate
#     g: gradient ของ neuron
#     x: input ที่เข้าสู่ neuron
#     dw_prev: ค่า delta weight จากรอบก่อนหน้า
#     """
#     dw_new = m * dw_prev + l * g * x
#     return dw_new
