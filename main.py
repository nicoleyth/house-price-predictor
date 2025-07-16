"""
Author: Nicole Huang
Date: 7/15/2025
Description: Main class for house price predictor.
"""

### IMPORTS ###
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Read excel file and load into dataframe
dataset = pd.read_excel("dataset/HousePricePrediction.xlsx")
# print(dataset.head(5))
print(dataset.shape) # dataframe dimensions

# Data Preprocessing (types of data)
obj = (dataset.dtypes == 'object') # dataset.dtypes - data type per col
object_cols = list(obj[obj].index) # names of true cols
print("Categorical variables:", len(object_cols))

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))

# Exploratory Data Analysis (EPA)
# Create a heatmap to analyze correlation between features
numerical_dataset = dataset.select_dtypes(include=['number'])

plt.figure(figsize=(12,6))
sb.heatmap(numerical_dataset.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)
plt.title('House Price Feature Heatmap')
plt.savefig('output/epa_results/feature_heatmap.jpg', bbox_inches='tight')

# Analyze categorical features
unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('Num of Unique values of Categorical Features')
plt.xticks(rotation=90)
sb.barplot(x=object_cols, y=unique_values)
plt.savefig('output/epa_results/categorical_feature_analysis.jpg', bbox_inches='tight')

plt.figure(figsize=(18,36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1

for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sb.barplot(x=list(y.index), y=y)
    index += 1

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.tight_layout()

plt.savefig('output/epa_results/categorical_feature_count.jpg')

