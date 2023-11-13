import pandas as pd
import re

df = pd.read_csv("qa-with-replies.csv", index_col=0)
vh_count = 0

def remove_venlig_hilsen(text):
    global vh_count
    pattern = r"Venlig hilsen|Med venlig hilsen|Bedste hilsner|De bedste hilsner|Mange hilsner|Venlige hilsner|Gode hilsner"
    match = re.search(pattern, text)
    if match:
        vh_count += 1
    return text[:match.start()] if match else text

# Function to apply to each row for "Kære" patterns
def remove_kaere(text):
    pattern = r"Kære Spørger[ ,\n]*\[[^\]]+\]|Kære Spørger[ ,\n]*\([^\)]+\)"
    match = re.search(pattern, text)
    if match:
        end_pattern = re.search(r"\.|\n", text[match.end():])
        return text[:match.start()] + text[match.end() + (end_pattern.end() if end_pattern else 0):]
    else:
        return text

# Apply the cleaning functions to each row
df["reply"] = df["reply"].apply(remove_venlig_hilsen).apply(remove_kaere)
df_reply = df[['reply']].copy()
df_reply['label'] = 1
df_reply.rename(columns={'reply': 'text'}, inplace=True)
df_answer = df[['answer']].copy()
df_answer['label'] = 0
df_answer.rename(columns={'answer': 'text'}, inplace=True)
df = pd.concat([df_reply, df_answer], ignore_index=True)
df.to_csv("self-generated-data.csv", index=False)
