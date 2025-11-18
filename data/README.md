# data

We use the following datasets from Hugging Face:

```python
import datasets

datasets.load_dataset("clip-benchmark/wds_mscoco_captions", split="test")

datasets.load_dataset("imagenet-1k", split="validation")
```

File `data.ipynb` creates the pointing game dataset in this directory.