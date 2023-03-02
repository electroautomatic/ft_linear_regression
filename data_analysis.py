import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data from file 'data.csv'
# (in the same directory that your python process is based)

df = pd.read_csv('data.csv')

# Let's calculate the correlation coefficient of input data

def correlation_coefficient(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum([xi * yi for xi, yi in zip(x, y)])
    sum_x_squared = sum([xi ** 2 for xi in x])
    sum_y_squared = sum([yi ** 2 for yi in y])

    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x_squared - sum_x ** 2) * (n * sum_y_squared - sum_y ** 2)) ** 0.5

    if denominator == 0:
        return 0.0
    else:
        return numerator / denominator

r = correlation_coefficient(df['km'], df['price'])
print("Correlation coefficient of input data:", r)
if r > 0.5 or r < -0.5:
    print("The correlation coefficient is high enough to use linear regression")
else:
    print("The correlation coefficient is too low to use linear regression")

plt.scatter(df['km'], df['price'])
plt.grid()
plt.show()


