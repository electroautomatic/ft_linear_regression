import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def getData(file):
    df = pd.read_csv(file)
    x = df['km'].values
    y = df['price'].values
    return x, y

x, y = getData('data.csv')

k , b = np.polyfit(x, y, 1)
print("k = ", k, "b = ", b)
plt.figure(1)
plt.scatter(x, y)
plt.plot(x, k * x + b, 'r')
plt.xlabel('km')
plt.ylabel('price')
plt.show()