import os
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from tqdm import tqdm
import time

MODEL = "gpt-3.5-turbo"

# Load dataframe
df = pd.read_csv("qa-with-prompts.csv")

# Add a new column to store replies
df["reply"] = None



def get_reply(prompt):
    try:
        completion = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(10)  # Wait for 10 seconds before retrying
        return get_reply(prompt)  # Retry


# Iterate through the dataframe
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    if pd.isna(row["reply"]):
        reply = get_reply(row["prompt"])
        df.at[index, "reply"] = reply
        # Save progress every 10 iterations
        if index % 10 == 0:
            df.to_csv("qa-with-replies.csv", index=False)

# Save final dataframe
df.to_csv("qa-with-replies.csv", index=False)
