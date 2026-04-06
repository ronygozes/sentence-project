import numpy as np
import pandas as pd


df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, np.nan, 9], [10, 11, 12]]), columns=['a', 'b', 'c'], index=[1, 3, 4, 12])

print(df)
print(df.to_dict(orient='index'))

