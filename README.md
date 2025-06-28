# Caste-and-Migration-Hate-Speech
Official implementation of Team EM-26's submission to the LT-EDI@LDK 2025 Shared Task on Caste and Migration Hate Speech Detection in Tamil-English code-mixed social media text.


This repository provides an end-to-end pipeline for benchmarking multiple models (mBERT, XLM-RoBERTa-base, XLM-RoBERTa-large, and a CNN baseline) for hate speech detection using PyTorch and HuggingFace Transformers.

## Features

- Supports data augmentation using `nlpaug`
- Implements custom Focal Loss for handling class imbalance
- Detailed logging, plots, and model comparison

## Setup

```bash
pip install -r requirements.txt
