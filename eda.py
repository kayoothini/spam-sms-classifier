import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

sns.set_style("whitegrid")

# 1. Spam vs Ham Bar Chart
plt.figure(figsize=(6, 4))
df['label'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Spam vs Ham Count')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig('spam_ham_count.png')
plt.show()

# 2. Pie Chart
plt.figure(figsize=(6, 6))
df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                  colors=['green', 'red'], startangle=90)
plt.title('Spam vs Ham Percentage')
plt.ylabel('')
plt.savefig('spam_ham_pie.png')
plt.show()

# 3. Message Length Analysis
df['length'] = df['message'].apply(len)

plt.figure(figsize=(10, 5))
df[df['label'] == 'ham']['length'].plot(kind='hist', bins=50,
                                         alpha=0.6, color='green', label='Ham')
df[df['label'] == 'spam']['length'].plot(kind='hist', bins=50,
                                          alpha=0.6, color='red', label='Spam')
plt.title('Message Length Distribution')
plt.xlabel('Length')
plt.legend()
plt.savefig('message_length.png')
plt.show()

print("=== Insights ===")
print(f"Average Ham length: {df[df['label']=='ham']['length'].mean():.0f} characters")
print(f"Average Spam length: {df[df['label']=='spam']['length'].mean():.0f} characters")