# Third-party dataset notices

The vulnerability scan loads some scenario datasets from external Hugging Face
repositories at runtime. Their original licenses and attribution requirements
are listed below.

## Do-Not-Answer

Loaded from `giskardai/do-not-answer-scenarios`, derived from the **Do-Not-Answer**
dataset by Wang et al.

- Source: https://huggingface.co/datasets/LibrAI/do-not-answer
- Data license: **CC-BY-NC-SA-4.0** (NonCommercial — the scan excludes this dataset
  when `commercial_use=True`)
- Source code license: Apache-2.0

```bibtex
@misc{wang2023donotanswer,
    author = {Wang, Yuxia and Li, Haonan and Han, Xudong and Nakov, Preslav and Baldwin, Timothy},
    title = {Do-Not-Answer: A Dataset for Evaluating Safeguards in LLMs},
    year = {2023},
    howpublished = {arXiv preprint arXiv:2308.13387},
}
```
