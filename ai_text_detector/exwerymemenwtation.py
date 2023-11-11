import json
from pathlib import Path
from typing import Counter
import spacy
import pandas as pd
from pelutils import Table
import hunspell
import language_tool_python


def get_daaaata():
    return pd.read_csv(Path("ai_text_detector") / "data" / "data.csv")

def get_val_dataaa():
    with open("examples.json", "r", encoding="utf-8") as file:
        return pd.DataFrame({"text": json.load(file)})

def mean_sentence_length(nlp, text: str) -> float:
    doc = nlp(text)
    sentence_lengths = [len(sentence) for sentence in doc.sents]
    return sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0


def sentence_length_variability(nlp, text: str) -> float:
    doc = nlp(text)
    sentence_lengths = [len(sentence) for sentence in doc.sents]
    mean_length = (
        sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    )
    variance = (
        sum((length - mean_length) ** 2 for length in sentence_lengths)
        / len(sentence_lengths)
        if sentence_lengths
        else 0
    )
    return variance**0.5


def word_count(_, text: str) -> int:
    return len(text.split())


def unique_word_count(_, text: str) -> int:
    return len(set(text.split()))


def average_word_length(_, text: str) -> float:
    words = text.split()
    return sum(len(word) for word in words) / len(words) if words else 0


def type_token_ratio(_, text: str) -> float:
    words = text.split()
    return len(set(words)) / len(words) if words else 0


def special_char_count(_, text: str) -> int:
    return sum(not char.isalnum() and not char.isspace() for char in text)


def pos_counts(nlp, text):
    doc = nlp(text)
    pos_counts = Counter(token.pos_ for token in doc)
    return dict(pos_counts)


def ner_counts(nlp, text):
    doc = nlp(text)
    ner_counts = Counter(ent.label_ for ent in doc.ents)
    return dict(ner_counts)


def dependency_features(nlp, text):
    doc = nlp(text)
    depths = [token.head.i - token.i for token in doc]
    avg_depth = sum(depths) / len(depths) if depths else 0
    return avg_depth


def sentence_complexity(nlp, text):
    doc = nlp(text)
    num_clauses = sum(1 for token in doc if token.dep_ in ["ccomp", "xcomp"])
    num_sentences = sum(1 for _ in doc.sents)
    return num_clauses / num_sentences if num_sentences > 0 else 0


def lemma_counts(nlp, text):
    doc = nlp(text)
    lemmas = Counter(token.lemma_ for token in doc)
    return len(lemmas)

spello = hunspell.HunSpell("/home/sorenmulli/Nextcloud/Andet/Software/LaTeX/EndLosung/dict-da-2.4/da_DK.dic", "/home/sorenmulli/Nextcloud/Andet/Software/LaTeX/EndLosung/dict-da-2.4/da_DK.aff")
tool = language_tool_python.LanguageToolPublicAPI('da-DK')
def spello_mode(nlp, text):
    return len(tool.check(text)) / len(text.split())
    # doc = nlp(text)
    # lemmas = [token.lemma_  for token in doc]
    # lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma.islower()]
    # spelled = [spello.spell(lemma) for lemma in lemmas]
    # return sum(spelled) / len(spelled)

def contains_newlines(nlp, text):
    return int("\n\n" in text)


def run_funnystuff():
    # df = get_daaaata()
    df = get_val_dataaa()
    # python -m spacy download da_core_news_sm
    nlp = spacy.load("da_core_news_sm")
    for thingymadingy in (
        "contains_newlines",
        "spello_mode",
        "mean_sentence_length",
        "sentence_length_variability",
        "word_count",
        "unique_word_count",
        "average_word_length",
        "type_token_ratio",
        "special_char_count",
        "pos_counts",
        "ner_counts",
        "dependency_features",
        "sentence_complexity",
        "lemma_counts",
    ):
        df[thingymadingy] = df["text"].apply(
            lambda s: globals().get(thingymadingy)(nlp, s)
        )
    for column in df.columns:
        if isinstance(df[column].iloc[0], dict):
            expanded_col = pd.json_normalize(df[column])
            df = pd.concat([df, expanded_col], axis=1).fillna(0)
            df.drop(column, axis=1, inplace=True)
    t = Table()
    for col in df.select_dtypes(include=["number"]):
        if col == "is_generated":
            continue
        row = [col]
        # for is_generated in 0, 1:
        #     subdf = df[df.is_generated == is_generated]
        #     row.append("%.2f ± %.1f" % (subdf[col].mean(), 2*subdf[col].std()/len(subdf)**0.5))
        row.append("%.1f ± %.1f" % (df[col].mean(), 2*df[col].std()/len(df)**0.5))
        t.add_row(row)
    print(t)
    df.to_csv("val_with_metrics.csv")


run_funnystuff() if __name__ == "__main__" else print("dingaling")
