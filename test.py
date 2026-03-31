import numpy as np
import pandas as pd


df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]), columns=['a', 'b', 'c'])
print(df)

print(df['b'].tolist())


