# KiYOUNG2 RE Solution

## Task Description

### KLUE Releation Extraction Task

* A task to classify the relationship between subject and object entities.

### Overview

* Datasets
  * The size of Train dataset: 32,470
  * The size of Test dataset: 7,765
  * The number of classes: 30
* Entity Types
  * Subject: `PER`, `ORG`
  * Object:  `PER`, `ORG`, `LOC`, `POH`, `DAT`, `NOH`

## Solution Description

|                     | Micro F1 Score | AUPRC  | Rank      |
| :-----------------: | :------------: | :----: | :-------: |
| Public Leaderboard  | 77.002         | 81.908 | 1st place |
| Private Leaderboard | 75.725         | 82.261 | 1st place |

> You can check our solution description : [PRESENTATION LINK](https://github.com/boostcampaitech2/klue-level2-nlp-14/blob/main/assets/kiyoung2_klue_re.pdf)

## Usage

### Requirements

```bash
git clone https://github.com/boostcampaitech2/klue-level2-nlp-14.git
cd klue-level2-nlp-14
pip install -r requirements.txt
```

### Getting Started

- run klue task

  ```bash
  python new_run.py configs/{YOUR_CONFIG}.yaml
  ```

- run hp search

  ```bash
  python new_hps.py configs/{YOUR_CONFIG}.yaml
  ```

- run ner module

  ```python
  >>> from solution.ner import NERInterface
  [Korean Sentence Splitter]: Initializing Kss..
  >>> ner = NERInterface.from_pretrained("ko")
  >>> ner(["이순신은 조선 중기의 무신이다", "오늘도 좋은 하루입니다. 기영이   화이팅입니다!"])
  [[('이순신', 'PERSON'), ('은', 'O'), (' ', 'O'), ('조선 중기', 'DATE'), ('의', 'O'), (' ', 'O'), ('무신', 'CIVILIZATION'), ('이다', 'O')], [('오늘', 'DATE'), ('도', 'O'), (' ', 'O'), ('좋은', 'O'), (' ', 'O'), ('하루', 'DATE'), ('입니다.', 'O'), (' ', 'O'), ('기영', 'PERSON'), ('이', 'O'), (' ', 'O'), ('화이팅입니다!', 'O')]]
  ```

- run k-fold ensemble
  - In the `run_kfold.sh` file, modify `CONFIG_DIR` to your yaml file path.
  - And then, run.
  
  ```bash
  chmod +x run_kfold.sh
  ./run_kfold.sh
  ```


## Reference
```
@inproceedings{lyu-chen-2021-relation,
  title = "Relation Classification with Entity Type Restriction",
  author = "Lyu, Shengfei  and
    Chen, Huanhuan",
  booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
  month = aug,
  year = "2021",
  address = "Online",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2021.findings-acl.34",
  doi = "10.18653/v1/2021.findings-acl.34",
  pages = "390--395",
}

@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}

@inproceedings{optuna_2019,
    title={Optuna: A Next-generation Hyperparameter Optimization Framework},
    author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
    booktitle={Proceedings of the 25rd {ACM} {SIGKDD} International Conference on Knowledge Discovery and Data Mining},
    year={2019}
}

@article{han2021ptr,
  title={PTR: Prompt Tuning with Rules for Text Classification},
  author={Han, Xu and Zhao, Weilin and Ding, Ning and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2105.11259},
  year={2021}

@article{liaw2018tune,
    title={Tune: A Research Platform for Distributed Model Selection and Training},
    author={Liaw, Richard and Liang, Eric and Nishihara, Robert and
            Moritz, Philipp and Gonzalez, Joseph E and Stoica, Ion},
    journal={arXiv preprint arXiv:1807.05118},
    year={2018}
}

@software{quentin_lhoest_2021_5510481,
  author       = {Quentin Lhoest and
                  Albert Villanova del Moral and
                  Patrick von Platen and
                  Thomas Wolf and
                  Yacine Jernite and
                  Abhishek Thakur and
                  Lewis Tunstall and
                  Suraj Patil and
                  Mariama Drame and
                  Julien Chaumond and
                  Julien Plu and
                  Joe Davison and
                  Simon Brandeis and
                  Teven Le Scao and
                  Victor Sanh and
                  Kevin Canwen Xu and
                  Nicolas Patry and
                  Angelina McMillan-Major and
                  Philipp Schmid and
                  Sylvain Gugger and
                  Steven Liu and
                  Nathan Raw and
                  Sylvain Lesage and
                  Théo Matussière and
                  Lysandre Debut and
                  Stas Bekman and
                  Clément Delangue},
  title        = {huggingface/datasets: 1.12.1},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {1.12.1},
  doi          = {10.5281/zenodo.5510481},
  url          = {https://doi.org/10.5281/zenodo.5510481}
}

@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation},
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jung-Woo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
