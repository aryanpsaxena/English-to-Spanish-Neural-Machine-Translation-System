# English-to-Spanish Neural Machine Translation: Transformer vs mT5 vs LLaMA vs Gemma

**Course:** Foundations of Generative AI (ICT4403), Assignment 3
**Student:** Aryan Saxena | **Reg. No.:** 220953552

> Fill every `<...>` from your own run (`outputs/*.csv`). Delete this note before submitting.

## 1. Problem Statement and Objective
Build English-to-Spanish translation systems with two encoder-decoder models (a Transformer trained from scratch, fine-tuned mT5) and two decoder-only models (LLaMA, Gemma), evaluate them on the same test set with BLEU, chrF, training time, inference time and parameter count, compare the architectures, and discuss responsible-AI concerns.

## 2. Dataset
| Item | Detail |
|---|---|
| Name | OPUS Books, English-Spanish (sentence-aligned public-domain book translations) |
| Source | https://huggingface.co/datasets/Helsinki-NLP/opus_books (original: https://opus.nlpl.eu/Books/corpus/version/Books) |
| Cleaning | NFC normalisation, whitespace collapse, 3-40 words, length ratio <= 2, no copies/duplicates |
| Subset and split | 10,000 pairs: 8,000 train / 1,000 validation / 1,000 test (random, seed 42) |
| Avg. length | `<EN words>` / `<ES words>` |

## 3. Models
| Model | Architecture | Adaptation | Parameters (trainable) |
|---|---|---|---|
| Transformer | encoder-decoder, 3+3 layers, d=256, joint 8k BPE | trained from scratch | `<fill>` |
| mT5-small (`google/mt5-small`) | encoder-decoder | full fine-tuning, 5 epochs | `<fill>` |
| LLaMA (`<model id used>`) | decoder-only | 3-shot prompting, then LoRA (r=16, 3,000 pairs, 1 epoch) | `<fill>` (`<LoRA params>`) |
| Gemma (`<model id used>`) | decoder-only | 3-shot prompting, then LoRA (r=16, 3,000 pairs, 1 epoch) | `<fill>` (`<LoRA params>`) |

All models: same test set, greedy decoding, batch size 32, max 80 new tokens.

## 4. Execution Steps
1. Open `FGAI_Assignment3_Machine_Translation.ipynb` in Google Colab, set `Runtime > Change runtime type > GPU`.
2. For the official LLaMA/Gemma weights, accept the licences on Hugging Face and add a Colab secret `HF_TOKEN` (otherwise ungated mirrors are used).
3. `Runtime > Run all` (about 60-90 minutes on a T4). Results are written to `outputs/` and zipped.

Dependencies: `torch transformers datasets peft sacrebleu sentencepiece accelerate pandas matplotlib` (installed by the first cell).

## 5. Results
| Model | BLEU | chrF | Train/FT time (s) | Inference (ms/sentence) | Params (M) | Trainable (M) |
|---|---|---|---|---|---|---|
| Transformer | | | | | | |
| mT5-small | | | | | | |
| LLaMA | | | | | | |
| Gemma | | | | | | |

![Comparison](outputs/model_comparison.png)

Prompting vs LoRA (first 200 test sentences): `<paste outputs/adaptation_comparison.csv>`

Sample translations: see `outputs/test_translations.csv`.

## 6. Model Comparison and Evaluation
`<Encoder-decoder vs decoder-only: quality, cost, size, pretraining, tokenization (tokens per Spanish word), adaptation method. 6-8 lines using your numbers.>`

## 7. Sampling-Temperature Analysis
Temperatures 0.2, 0.7, 1.2 vs greedy on the first 100 test sentences (`outputs/temperature_analysis.csv`).

![Temperature](outputs/temperature_analysis.png)

`<Summarise the BLEU/chrF drop and repetition trend per model.>`

## 8. Responsible and Ethical AI
Automatic risk screening (`outputs/risk_screening.csv`) and targeted probes (`outputs/ethics_probes.csv`).
- **Bias:** `<gender stereotypes seen in probes 1, 2, 8>`
- **Hallucination/omission:** `<examples and flag rates>`
- **Cultural misrepresentation:** `<idiom probes 3, 4, 10>`
- **Sensitive content:** `<medical/legal probes 5, 6, 8, 9>`
- **Mitigations:** human review for high-stakes text, bias-aware evaluation sets, balanced data, low-temperature decoding with consistency checks, model cards and transparency, privacy safeguards.

## 9. Conclusion
**Best-performing model: `<name>`.** `<3-4 lines justifying with BLEU/chrF, cost, and trade-offs.>`

## 10. Repository Structure
```
.
├── FGAI_Assignment3_Machine_Translation.ipynb
├── README.md
└── outputs/   (results_table.csv, adaptation_comparison.csv, temperature_analysis.csv,
                risk_screening.csv, ethics_probes.csv, test_translations.csv, final_ranking.csv, *.png)
```

