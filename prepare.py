import json

from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from langdetect import detect


stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()


def read_amazon_format(path: str, column_name: str, num_review=None, remove_stop_words=False):
    """
    Подготовка данных из json файла

    :param path: путь к json файлу
    :param column_name: имя поля с текстом отзыва
    :param num_review: количество отбираемых отзывов (по умолчанию все)
    :param remove_stop_words: флаг – удаление стоп слов
    """
    new_path = path.split('.')[0] + '.txt'
    with open(new_path, "w+", encoding="utf-8") as wf:

        for line in open(path, "r", encoding="utf-8"):
            reviews = list(json.loads(line.strip())[column_name].values())
            num_review = len(reviews) if num_review is None or num_review >= len(reviews) else num_review


                tokenized_sentences = [word_tokenize(sentence) for sentence in sentences if len(sentence.split()) >= 2]

                if remove_stop_words:
                    lemmatized_sentences = [[morph.normal_forms(word)[0] for word in s if not word in stopwords_ru and str.isalpha(word)]
                                            for s in tokenized_sentences]
                else:
                    lemmatized_sentences = [[morph.normal_forms(word)[0] for word in s if str.isalpha(word)]
                                            for s in tokenized_sentences]

                lemmatized_sentences = [sent for sent in lemmatized_sentences if sent != []]

                for sentence in lemmatized_sentences:
                    wf.write(" ".join(sentence) + "\n" if sentence else " ")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        path = sys.argv[1]
        column_name = sys.argv[2]
        n_review = int(sys.argv[3]) if len(sys.argv) > 3 else None
        read_amazon_format(path, column_name, n_review, remove_stop_words=True)
    else:
        print("""
            Пожалуйста, введите:
                1. Имя файла
                2. Имя поля, в которой записаны отзывы
                3. Количество отзывов на выходе
        """)
