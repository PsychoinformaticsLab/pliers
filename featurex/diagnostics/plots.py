import pandas as pd
import seaborn as sns

from featurex.diagnostics.collinearity import correlation_matrix

df = pd.DataFrame.from_csv('/Users/quinnmac/Documents/test_df.csv')
sns.heatmap(correlation_matrix(df), cmap='Blues')
sns.plt.show()