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

> You can check our solution description : [PRESENTATION LINK](https://github.com/jinmang2/boostcamp_ai_tech_2/blob/main/assets/ppt/klue_re.pdf)

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
- [Ray Tune - pbt_transformers_example](https://docs.ray.io/en/master/tune/examples/pbt_transformers.html)

```
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation},
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jung-Woo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
