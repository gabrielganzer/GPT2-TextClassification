"""Text processing pipeline."""
import json
import re
import string

import emoji
from bs4 import BeautifulSoup

MAPPING_PATH = r"../dataset/mapping.json"
TRAIN_PATH = r"../dataset/train.csv"
TEST_PATH = r"../dataset/test.csv"

file = open(MAPPING_PATH)
mapping = json.load(file)
file.close()


def clean_text(text):
    """Clean emoji, Make text lowercase, remove text in square brackets, \
    remove links,remove punctuation and remove words containing numbers."""
    text = emoji.demojize(text)
    text = re.sub(r"\:(.*?)\:", "", text)
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = BeautifulSoup(text, "lxml").get_text()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
    return text


def clean_contractions(text, mapping):
    """Clean contractions."""
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    for word in mapping.keys():
        if "" + word + "" in text:
            text = text.replace("" + word + "", "" + mapping[word] + "")
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text


def clean_special_chars(text, punct, mapping):
    """Clean special characters present."""
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f" {p} ")

    specials = {
        "\u200b": " ",
        "…": " ... ",
        "\ufeff": "",
        "करना": "",
        "है": "",
    }
    for s in specials:
        text = text.replace(s, specials[s])

    return text


def correct_spelling(x, dic):
    """Correct common spelling errors."""
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x


def remove_space(text):
    """Remove spaces."""
    text = text.strip()
    text = text.split()
    return " ".join(text)


def text_preprocessing(text):
    """Clean and parse the text."""
    text = clean_text(text)
    text = clean_contractions(text, mapping["contraction_mapping"])
    text = clean_special_chars(
        text, mapping["punct"], mapping["punct_mapping"]
    )
    text = correct_spelling(text, mapping["mispell_dict"])
    text = remove_space(text)

    return text
