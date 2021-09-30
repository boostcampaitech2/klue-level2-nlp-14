import os
import json
import csv
import datasets


_CITATION = """\
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation},
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""
_LICENSE = "CC-BY-SA-4.0"
_DESCRIPTION = """\
KLUE (Korean Language Understanding Evaluation)
Korean Language Understanding Evaluation (KLUE) benchmark is a series of datasets to evaluate natural language
understanding capability of Korean language models. KLUE consists of 8 diverse and representative tasks, which are accessible
to anyone without any restrictions. With ethical considerations in mind, we deliberately design annotation guidelines to obtain
unambiguous annotations for all datasets. Futhermore, we build an evaluation system and carefully choose evaluations metrics
for every task, thus establishing fair comparison across Korean language models.
"""
_URL = ""
_DATA_URL = "/opt/ml/dataset/" # LOCAL FILE


class KlueREConfig(datasets.BuilderConfig):

    def __init__(self, data_url, features, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.data_url = data_url
        self.features = features


class KlueRE(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        KlueREConfig(
            name="re",
            data_url=_DATA_URL,
            description="",
            features=datasets.Features(
                {
                    "guid": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "subject_entity": {
                        "word": datasets.Value("string"),
                        "start_idx": datasets.Value("int32"),
                        "end_idx": datasets.Value("int32"),
                        "type": datasets.Value("string"),
                    },
                    "object_entity": {
                        "word": datasets.Value("string"),
                        "start_idx": datasets.Value("int32"),
                        "end_idx": datasets.Value("int32"),
                        "type": datasets.Value("string"),
                    },
                    "label": datasets.ClassLabel(
                        names=[
                            "no_relation",
                            "org:top_members/employees",
                            "org:members",
                            "org:product",
                            "per:title",
                            "org:alternate_names",
                            "per:employee_of",
                            "org:place_of_headquarters",
                            "per:product",
                            "org:number_of_employees/members",
                            "per:children",
                            "per:place_of_residence",
                            "per:alternate_names",
                            "per:other_family",
                            "per:colleagues",
                            "per:origin",
                            "per:siblings",
                            "per:spouse",
                            "org:founded",
                            "org:political/religious_affiliation",
                            "org:member_of",
                            "per:parents",
                            "org:dissolved",
                            "per:schools_attended",
                            "per:date_of_death",
                            "per:date_of_birth",
                            "per:place_of_birth",
                            "per:place_of_death",
                            "org:founded_by",
                            "per:religion"
                        ]
                    ),
                    "source": datasets.Value("string"),
                },
            )
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            homepage=_URL,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.download_manager.DownloadManager):
        """ Returns SplitGenerators. """
        # data_file = dl_manager.download_and_extract(self.config.data_url)
        data_file = self.config.data_url
        return [
            datasets.SplitGenerator(
                # name=datasets.Split.TRAIN,
                name="train",
                gen_kwargs={
                    "data_file": os.path.join(data_file, "train/train.csv"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                # name=datasets.Split.VALIDATION,
                name="test",
                gen_kwargs={
                    "data_file": os.path.join(data_file, "test/test_data.csv"),
                    "split": "test",
                },
            )
        ]

    def _generate_examples(self, data_file: str, split: str):
        """ Yields examples. """
        with open(data_file, newline='', encoding="UTF-8") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            feature_names = next(reader)
            # assert all(key in feature_names for key in self.config.features)
            for ix, row in enumerate(reader):
                features = {
                    "guid": row[0],
                    "sentence": row[1],
                    "subject_entity": eval(row[2]),
                    "object_entity": eval(row[3]),
                    "label": row[4] if row[4] != "100" else "no_relation",
                    "source": row[5],
                }
                yield ix, features