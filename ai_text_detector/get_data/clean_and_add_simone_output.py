import pandas as pd
df = pd.read_csv("local-data/simone-qar.csv")
df["Answer"] = df["Answer"].apply(lambda s: " ".join(s.replace("\n", "").split()))
df["AI Answer"] = df["AI Answer"].apply(lambda s: " ".join(s.replace("\n", "").split()))

df_reply = df[["AI Answer"]].copy()
df_reply["label"] = 1
df_reply.rename(columns={"AI Answer": "text"}, inplace=True)

df_answer = df[["Answer"]].copy()
df_answer["label"] = 0
df_answer.rename(columns={"Answer": 'text'}, inplace=True)
df = pd.concat([df_reply, df_answer], ignore_index=True)

out = "self-generated-data.csv"
existing_df = pd.read_csv(out)
df = pd.concat([existing_df, df])
df.to_csv("self-generated-data.csv", index=False)
