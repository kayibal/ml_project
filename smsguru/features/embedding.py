import luigi
import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from smsguru import settings
from smsguru.features.clean import *
from smsguru.features.tasks import AddCategoryLabels
from smsguru.hdf5 import Hdf5TableTarget


def create_train_dataset(questions, model):
    res = []
    for idx, pid, cid, question in zip(questions.question_id, questions.parent_id, questions.category_id, questions.question):
        wordlist = clean_language(question)
        matrix = np.empty((len(wordlist), 304))
        matrix[:, 0] = idx
        matrix[:, 1] = pid
        matrix[:, 2] = cid
        matrix[:, 3] = np.arange(len(wordlist))
        for row, w in enumerate(wordlist):
            try:
                matrix[row, 4:] = model[w]
            except KeyError:
                pass
        train_example = pd.DataFrame(matrix)
        train_example.columns = ["question_id","parent_id",  "category_id", "wordno"] \
                                + ["dim_{}".format(i) for i in range(300)]
        res.append(train_example)

    return pd.concat(res, axis=0)

_reduce_methods = {
    "mean": pd.Series.mean,
    "sum": pd.Series.sum,
    "median": pd.Series.median,
    "full": 0,
}

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

class EmbeddingDataset(luigi.Task):

    reduce_method = luigi.ChoiceParameter(choices=list(_reduce_methods.keys()))
    outfile = luigi.Parameter(default=None)
    model = luigi.Parameter(default=os.path.join(settings.PROJECT_DIR, "german.model"))

    def requires(self):
        return AddCategoryLabels()

    def output(self):
        if self.outfile is None:
            self.outfile = "/srv/smsguru/dataset/embedded_{}.hdf5".format(self.reduce_method)
        return Hdf5TableTarget(self.outfile, "/df")

    def run(self):
        w2v = Word2Vec.load_word2vec_format(self.model, binary=1)
        questions = self.input().open().read(autoclose=1).reset_index()
        with self.output().open("w") as table:
            table.write(create_train_dataset(questions, w2v))