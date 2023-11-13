import pandas as pd
df = pd.read_csv("questions_answers.csv")
print(len(df))
df = df[df['answer'].str.split().str.len() >= 3]
df = df.map(lambda x: x.strip() if isinstance(x, str) else x).drop_duplicates(subset="question").drop_duplicates(subset="answer").dropna()

print(len(df))
df.to_csv("qa-cleaned.csv")
