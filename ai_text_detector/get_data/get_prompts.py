import numpy as np
import pandas as pd
df = pd.read_csv("qa-cleaned.csv", index_col=0)
prompts = []
for index, row in df.iterrows():
    question = row["question"]
    source = row["source"]
    answers = df[(df["source"] == source) & (df["answer"] != row["answer"])]["answer"]
    chosen_answer = np.random.choice(answers, 1)[0]
    corresponding_question = df[df["answer"] == chosen_answer].question.iloc[0]
    prompts.append(
            "Du skal svare på et spørgsmål, hvor du skal lege, at du er en dansk ekspert, politiker eller kendis. Svar KUN på spørgsmålet.\n" +
            "Du kan lade dig inspirere at følgende eksempel\n" +
            "# Spørgsmål (eksempel)\n%s\n# Svar (eksempel)\n%s\n# Spørgsmål\n%s\n# Svar" % (corresponding_question, chosen_answer, question)
    )
df["prompt"] = prompts
for prompt in df.prompt.sample(10):
    print(prompt)
    print("-"*100)
df.to_csv("qa-with-prompts.csv")
