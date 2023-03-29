# -*- coding: utf-8 -*-

import codecs
import sys

import gensim
from tqdm import tqdm


class Sentences(object):
    def __init__(self, filename: str):
        self.filename = filename

    def __iter__(self):
        for line in tqdm(codecs.open(self.filename, "r", encoding="utf-8"), self.filename):
            yield line.strip().split()


def main(path):
    sentences = Sentences(path)
    model = gensim.models.Word2Vec(sentences, vector_size=200, window=5, min_count=5, workers=7, sg=1,
                                   negative=5, max_vocab_size=20000)
    model.save("word_vectors/" + path + ".w2v")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        raise Exception('Введите имя файла')

    try:
        import os
        os.mkdir("word_vectors/")
    except:
        pass

    print("Training w2v on dataset", path)
    main(path)
    print("Training done.")
