import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas import read_excel

# get the input xls file 
measurements_file = read_excel("measurements.xlsx", sheet_name="Sheet2")

# seaborn styling
sns.set()

# plot the matrix N to CPU time
plt.plot(measurements_file["N"], measurements_file["cpu time"], "rs-")
plt.plot(measurements_file["N"], measurements_file["gpu time (Cores 512)"], "bo-")
plt.plot(measurements_file["N"], measurements_file["gpu time (Cores 256)"], "g^-")
# plt.xscale("log")
plt.xlabel("Matrix row /column size")
plt.ylabel("Time (milliseconds)")
plt.legend(loc="upper left")
plt.show()