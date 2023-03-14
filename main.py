import sys
import os

import spacy
import pandas as pd
import nltk
from nltk.corpus import stopwords
from spacy.symbols import NOUN, PROPN, PUNCT, SYM, ADP, DET

nlp = spacy.load("ru_core_news_lg")
stopwords_ru = stopwords.words("russian")


def check_digit(s):
    return '1' in s or '2' in s or '3' in s or '4' in s or '5' in s or '6' in s or '7' in s or '8' in s or '9' in s \
           or '0' in s or '(' in s or ')' in s or ':' in s or '+' in s or '!' in s or '"' in s or '-' in s


def tokenize(sentences):
    """
    Токенизация по словам
    """
    tokens = [x.lemma_ for senten in sentences for x in senten if len(x) > 1 and
              not check_digit(str(x)) and not str(x) in stopwords_ru]
    return tokens


def noun_chunks(obj):
    """
    Извлечение именных групп
    """
    labels = [
        "nsubj",
        "dobj",
        "nsubjpass",
        "pcomp",
        "pobj",
        "dative",
        "appos",
        "attr",
        "ROOT",
    ]
    res = []
    doc = obj.doc
    np_deps = [doc.vocab.strings.add(label) for label in labels]
    conj = doc.vocab.strings.add("conj")
    seen = set()
    for i, word in enumerate(obj):
        if word.pos not in (NOUN, PROPN):
            continue
        if word.i in seen:
            continue
        if word.dep in np_deps or True:
            if any(w.i in seen for w in word.subtree):
                continue
            seen.update(j for j in range(word.left_edge.i, word.i + 1))
            res.append([word.left_edge.i, word.i + 1])
        elif word.dep == conj:
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head
            if head.dep in np_deps:
                if any(w.i in seen for w in word.subtree):
                    continue
                seen.update(j for j in range(word.left_edge.i, word.i + 1))
                res.append([word.left_edge.i, word.i + 1])
    return res


def noun_find(sent_list):
    """
    Поиск существительных и именных групп
    """
    res = []
    for nlp_sent in sent_list:
        a = []
        num = noun_chunks(nlp_sent)
        for i in num:
            res_str = ''
            # именные группы
            for j in range(i[0], i[1]):
                if nlp_sent[j].pos in (PUNCT, SYM, ADP, 86, DET) or str(nlp_sent[j]) in stopwords_ru or check_digit(
                        str(nlp_sent[j])):
                    continue
                if res_str != '':
                    res_str += ' '
                res_str += str(nlp_sent[j].lemma_)
            if res_str != '':
                a.append(res_str)

        res.append(a)
    return res


def sent_token(text):
    """
    Токенизация по предложениям
    """
    text = str(text).lower()
    text = text.replace('.', '. ')
    return [nlp(x) for x in nltk.sent_tokenize(text)]


def process(df):
    """
    Основной процесс
    """
    df["sentences"] = df["text"].progress_map(lambda x: sent_token(x))
    df["noun_chunks"] = df["sentences"].progress_map(lambda x: noun_find(x))


if __name__ == '__main__':
    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        raise Exception("File doesn't exist!")

    df = pd.read_excel('review.xlsx')
    df = df[~df['text'].isnull()]

    process(df)

