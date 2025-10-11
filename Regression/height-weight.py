import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("../AI with Python/weight-height.csv", skiprows=1,names=["x","y"] )

print(df.corr())