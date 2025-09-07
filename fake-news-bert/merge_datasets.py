import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake['label'] = 'FAKE'
true['label'] = 'REAL'

df = pd.concat([fake, true])
df.to_csv("news.csv", index=False)
