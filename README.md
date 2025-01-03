# Information Fusion DPO

This repository contains the implementation of our paper "A Novel Alignment Approach based on Information Fusion View: Direct Preference Optimization with Dynamic Learnable β".

## Overview

We propose a novel theoretical framework that views Direct Preference Optimization (DPO) as an information fusion process between the reference model and human preference signals. Based on this view, we introduce a learnable and dynamically adjusted β parameter that allows the model to automatically balance between preserving reference knowledge and incorporating human feedback.

## Key Features

- Information fusion perspective on DPO
- Dynamic learnable β parameter
- Improved theoretical interpretability
- Superior performance across various models and datasets

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Training script:
```bash
python train.py --config config/default.yaml
```

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{info-fusion-dpo,
  title={A Novel Alignment Approach based on Information Fusion View: Direct Preference Optimization with Dynamic Learnable β},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## Acknowledgements

This code is based on the [beta-DPO](https://github.com/junkangwu/beta-DPO) implementation. 