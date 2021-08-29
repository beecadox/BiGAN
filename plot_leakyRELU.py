from matplotlib import pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
def leakyrelu(x, alpha=0.01):
    if x > 0:
        return np.maximum(0, x)
    else:
        return x * alpha

x = np.arange(-10, 10, 0.2)
y = [leakyrelu(i, alpha=0.2) for i in x]
plt.axvline(x=0, color="black", linewidth=.5)
plt.axhline(y=0, color="black", linewidth=.5)
plt.plot(x, y, color="#91bb61")
plt.title("LeakyReLU. Alpha=0.2")
plt.show()
