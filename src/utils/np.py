import numpy as np
import time

n = 10_000_000  # 1000万

# 纯 Python
start = time.time()
total = sum(i * i for i in range(n))
print(f"Python: {time.time() - start:.3f}s")

# NumPy
arr = np.arange(n)
start = time.time()
total = np.sum(arr ** 2)
print(f"NumPy: {time.time() - start:.3f}s")

