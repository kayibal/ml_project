import os

import luigi

from smsguru import settings
from smsguru.hdf5 import Hdf5TableTarget


def add_parent_labels(questions, categories):
    """
    Add parent labels from the categories dataset
    :param questions:
    :param categories:
    :return:
    """
    questions = questions[questions.category_main_id != 'N']
    questions["category_id"] = questions.category_main_id.astype(int)
    questions.drop("category_main_id", axis=1, inplace=1)
    questions = questions.merge(categories, how="left").set_index("question_id")
    return questions


class DataSource(luigi.ExternalTask):

    data_file = luigi.Parameter()

    def output(self):
        print(os.path.join(settings.PROJECT_DIR, self.data_file))
        return Hdf5TableTarget(os.path.join(settings.PROJECT_DIR, self.data_file), "/df")

class AddCategoryLabels(luigi.Task):

    category_cols = luigi.ListParameter(default=["category_id", "parent_id"])
    question_cols = luigi.ListParameter(default=["category_main_id", "question_id", "question", "tags"])
    outfile = luigi.Parameter(default="/srv/smsguru/merged_questions.hdf5")

    def output(self):
        return Hdf5TableTarget(self.outfile, "/df")

    def requires(self):
        return {"questions":DataSource(data_file="data/question_train.hdf5"),
                "categories": DataSource(data_file="data/category.hdf5")}

    def run(self):
        categories = self.input()["categories"].open().read(autoclose=1)[list(self.category_cols)]
        questions = self.input()["questions"].open().read(autoclose=1)[list(self.question_cols)]
        with self.output().open("w") as table:
            table.write(add_parent_labels(questions, categories))