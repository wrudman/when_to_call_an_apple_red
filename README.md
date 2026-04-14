# When to Call an Apple Red: Humans Follow Introspective Rules, VLMs Don't

> **Jonathan Nemitz¹, Carsten Eickhoff¹, Jessy Li², Kyle Mahowald², Michal Golovanevsky³\*, William Rudman²\***
>
> ¹University of Tübingen · ²The University of Texas at Austin · ³Brown University
---
<img width="859" height="406" alt="sample" src="https://github.com/user-attachments/assets/2c21b590-1d0f-45bc-b613-5187a6b2924d" />


## Overview

This repository contains the **Graded Color Attribution (GCA)** dataset and evaluation code for our paper studying whether Vision-Language Models (VLMs) faithfully follow their own introspective reasoning rules. GCA is available on HuggingFace: mgolov/graded-color-attribution. Use run_eval.py to get results for VLMs on GCA. 

**Key finding:** VLMs systematically violate their own stated reasoning rules when world-knowledge priors are present — a failure mode that does *not* mirror human cognition. Humans remain faithful to their stated rules; models do not.

---

## The GCA Dataset

GCA is a controlled benchmark consisting of black-and-white line drawings that vary pixel-level color coverage across three conditions:

| Condition | Description |
|---|---|
| **Prior-aligned** | Objects recolored with their canonical color (e.g., a red apple) |
| **Counterfactual** | Same objects recolored with an unexpected color (e.g., a blue strawberry) |
| **No-prior** | Abstract shapes with no color associations (e.g., a yellow pentagon) |

Color coverage is controlled by a threshold τ ∈ {0, 5, 10, 20, 30, 40, 50, 55, 60, 70, 80, 90, 100}%, allowing precise measurement of how visual evidence interacts with semantic priors.

The dataset contains:
- **220** base object categories
- **25** abstract shape outlines
- **~19,000** total image variants
- Human evaluation subset: **3,003** unique image variants across **173** participants

---

## Key Results

### VLMs violate their own rules
- GPT-4o-mini violates its stated introspective rules in **~60% of cases** on objects with strong color priors
- Lower-capacity models (Claude Haiku 4.5, Qwen 3.5-9B) show faithfulness rates as low as **20%**
- Failures occur even though frontier models accurately estimate pixel color proportions

### Humans are faithful
- Human faithfulness remains **~80%** when evaluated against empirically derived thresholds
- Any apparent violations are explained by a well-documented tendency to **overestimate color coverage**, not by a disconnect between stated rules and behavior
- Human responses are **consistent across all three stimulus types**

### World-knowledge priors are the culprit
- Faithfulness degrades for objects with strong priors regardless of whether recoloring is aligned *or* counterfactual
- Object identity alone is sufficient to activate a canonical color prior and override stated reasoning
- This effect is **absent for abstract shapes**, which carry no color associations

---

## Models Evaluated

- GPT-4o-mini
- Claude Opus 4.6
- Claude Haiku 4.5
- Qwen 3.5-9B

---

## Repository Structure

```
gca/
├── data/
│   ├── images/              # GCA image stimuli
│   │   ├── prior_aligned/
│   │   ├── counterfactual/
│   │   └── no_prior/
│   └── metadata.csv         # Object labels, thresholds, and categories
├── evaluation/
│   ├── vlm_eval.py          # VLM evaluation pipeline
│   ├── human_eval/          # Human trial survey structure and results
│   └── cot_prompts.py       # CoT prompt variants (Standard, Visual-Prior, Post-hoc, Text-Prior)
├── analysis/
│   ├── faithfulness.py      # Faithfulness and consistency metrics
│   └── figures.py           # Figure generation scripts
└── README.md
```

---

## Getting Started

### Installation

```bash
git clone https://github.com/<your-org>/gca.git
cd gca
pip install -r requirements.txt
```

### Running VLM Evaluation

```bash
python evaluation/vlm_eval.py \
  --model gpt-4o-mini \
  --stimulus_type prior_aligned \
  --cot_strategy standard
```

Available `--cot_strategy` options: `standard`, `visual_prior`, `post_hoc`, `text_prior`

Available `--stimulus_type` options: `prior_aligned`, `counterfactual`, `no_prior`

### Computing Faithfulness Metrics

```bash
python analysis/faithfulness.py --results_dir outputs/
```

---

## Citation

If you use GCA in your research, please cite:

```bibtex
@article{nemitz2025apple,
  title={When to Call an Apple Red: Humans Follow Introspective Rules, VLMs Don't},
  author={Nemitz, Jonathan and Eickhoff, Carsten and Li, Jessy and Mahowald, Kyle and Golovanevsky, Michal and Rudman, William},
  journal={arXiv preprint},
  year={2025}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgments

Object categories and color associations are drawn from the [Visual CounterFact](https://github.com/michal-golovanevsky/visual-counterfact) dataset. Human participant recruitment was conducted via [Prolific](https://www.prolific.com). This work was supported by the University of Tübingen, The University of Texas at Austin, and Brown University.
