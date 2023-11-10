from pathlib import Path
from typing import Counter
import spacy
import pandas as pd
from pelutils import Table


def get_daaaata():
    return pd.read_csv(Path("data") / "data.csv")


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
    return pos_counts


def ner_counts(nlp, text):
    doc = nlp(text)
    ner_counts = Counter(ent.label_ for ent in doc.ents)
    return ner_counts


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


def run_funnystuff():
    df = get_daaaata()
    # python -m spacy download da_core_news_sm
    nlp = spacy.load("da_core_news_sm")
    for thingymadingy in (
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

    t = Table()
    t.add_header(["Thingy", "🗿", "🤖"])
    for col in df.select_dtypes(include=["number"]):
        if col == "is_generated":
            continue
        row = [col]
        for is_generated in 0, 1:
            subdf = df[df.is_generated == is_generated]
            row.append("%.1f ± %.1f" % (subdf[col].mean(), 2*subdf[col].std()/len(df)**0.5))
        t.add_row(row)
    print(t)


run_funnystuff() if __name__ == "__main__" else print("dingaling")
