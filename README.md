# KiYOUNG2 RE Solution
KLUE Releation Extraction Tasks

## Usage

- run klue task

```
python new_run.py configs/{YOUR_CONFIG}.yaml
```

- run hp search

```
python new_hps.py configs/{YOUR_CONFIG}.yaml
```

- run ner module

```python
>>> from solution.ner import NERInterface
[Korean Sentence Splitter]: Initializing Kss..
>>> ner = NERInterface.from_pretrained("ko")
>>> ner(["이순신은 조선 중기의 무신이다", "오늘도 좋은 하루입니다. 기영이 화이팅입니다!"])
[[('이순신', 'PERSON'), ('은', 'O'), (' ', 'O'), ('조선 중기', 'DATE'), ('의', 'O'), (' ', 'O'), ('무신', 'CIVILIZATION'), ('이다', 'O')], [('오늘', 'DATE'), ('도', 'O'), (' ', 'O'), ('좋은', 'O'), (' ', 'O'), ('하루', 'DATE'), ('입니다.', 'O'), (' ', 'O'), ('기영', 'PERSON'), ('이', 'O'), (' ', 'O'), ('화이팅입니다!', 'O')]]

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
