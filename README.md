# NLP Cross-Variety Analysis (BESSTIE Dataset)

## Overview
This project explores Natural Language Processing techniques for sentiment classification and sarcasm detection across multiple English varieties (UK, AU, IN).

## Objectives
- Compare classical vs transformer models
- Analyse linguistic variation across dialects
- Evaluate cross-variety generalisation

## Methods
- TF-IDF + Logistic Regression (baseline)
- RoBERTa-base (transformer)
- Evaluation using Macro-F1

## Key Findings
- Classical models perform well on sentiment but fail on sarcasm
- Transformer models significantly improve performance
- Contextual understanding is critical for sarcasm detection

## Structure
- `notebooks/` → experiments
- `data/` → dataset
- `results/` → outputs and metrics

## How to Run
1. Install dependencies:

2. Run notebooks in order:
- 01 → dataset analysis
- 02 → baseline
- 03 → transformer

## Author
Reshabh Mehta , Mir Ishaque ali, Anil Emanuel ,Priyankya Chaterjee , Nakul Arora
MSc Artificial Intelligence, University of Surrey