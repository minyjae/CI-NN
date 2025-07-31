import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)  # y คือ output ของ sigmoid(x)

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
