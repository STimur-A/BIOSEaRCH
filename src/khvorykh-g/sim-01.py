"""
Simulation data for linear regression.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Intitiate variables
b0 = 70
b1 = 2.5
k = 5
n_samples = 1000
np.random.seed(42)

# Generate data
bmi = np.random.uniform(10, 35, n_samples)
y = b0 + b1*bmi + k*np.random.randn(n_samples)

# Create dataframe
data = pd.DataFrame({'bmi' : bmi, 'y': y})
print(data)

# Visualize data
plt.figure(figsize=(10,8))
plt.scatter(bmi,y)
plt.xlabel("BMI", fontsize = 14)
plt.ylabel("Disease progression", fontsize = 14)
plt.savefig("pics/sim-01.png")

# Save data
data.to_csv("out6/data-01.csv", index=False)