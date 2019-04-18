import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas import read_excel

# get the input xls file 
measurements_file = read_excel("measurements.xlsx")

# seaborn styling
sns.set()

# plot the matrix N to CPU time
plt.plot(measurements_file["N"], measurements_file["cpu time"]/1000, "ro-")
plt.plot(measurements_file["N"], measurements_file["gpu time"]/1000, "bo-")
plt.xlabel("Matrix row /column size")
plt.ylabel("Time (milliseconds)")
plt.show()˝