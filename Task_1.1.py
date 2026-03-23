from datasets import load_dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = load_dataset("surrey-nlp/BESSTIE-CW-26")

print(dataset)

train_df = pd.DataFrame(dataset['train'])
val_df = pd.DataFrame(dataset['validation'])

df = pd.concat([train_df, val_df])

# Sentiment overall
df['Sentiment'].value_counts().plot(kind='bar')
plt.title("Overall Sentiment Distribution")
plt.show()

# Sarcasm overall
df['Sarcasm'].value_counts().plot(kind='bar')
plt.title("Overall Sarcasm Distribution")
plt.show()

# Sentiment by variety
sns.countplot(x='variety', hue='Sentiment', data=df)
plt.title("Sentiment across varieties")
plt.show()

# Sarcasm by variety
sns.countplot(x='variety', hue='Sarcasm', data=df)
plt.title("Sarcasm across varieties")
plt.show()