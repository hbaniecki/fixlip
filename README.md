<div align="center">
  
# FIxLIP [NeurIPS 2025]
  
[![Conference](http://img.shields.io/badge/NeurIPS-2025-FFD93D.svg)](https://neurips.cc/Conferences/2025)
[![arXiv](http://img.shields.io/badge/Paper-arxiv.2508.05430-FF6B6B.svg)](https://arxiv.org/abs/2508.05430)
</div>

This repository is a code supplement to the following paper:

H. Baniecki, M. Muschalik, F. Fumagalli, B. Hammer, E. Hüllermeier, P. Biecek. **Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions**. NeurIPS 2025 https://openreview.net/forum?id=on22Rx5A4F

**TL;DR:** We introduce faithful interaction explanations of CLIP and SigLIP models (FIxLIP), offering a unique, game-theoretic perspective on interpreting image–text similarity predictions.

![https://openreview.net/forum?id=on22Rx5A4F](assets/figure1.png)

## Setup

```bash
conda env create -f env.yml
conda activate fixlip
```

## Start: example

`example.ipynb`

TODO: code chunk

TODO: add figure

## Details: experiments

* `src` - main code base with the FIxLIP implementation
* `data` - code for processing datasets
* `experiments` - code for running experiments
* `results` - experimental results
* `analysis` - analyze and visualize the results
* `gradeclip` - code and experiments with Grad-ECLIP
* `exclip` - code and experiments with exCLIP

## Citation

```bibtex
@inproceedings{baniecki2025explaining,
    title     = {Explaining Similarity in Vision-Language Encoders 
                 with Weighted Banzhaf Interactions},
    author    = {Hubert Baniecki and Maximilian Muschalik and Fabian Fumagalli and 
                 Barbara Hammer and Eyke H{\"u}llermeier and Przemyslaw Biecek},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2025},
    url       = {https://openreview.net/forum?id=on22Rx5A4F}
}
```

## Acknowledgements

This work was financially supported by the state budget within the Polish Ministry of Science and Higher Education program "Pearls of Science" project number PN/01/0087/2022.