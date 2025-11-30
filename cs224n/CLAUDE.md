# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Stanford CS 224N course repository containing assignments (a1-a4) and a GPT-2 implementation project. The main focus is on the `public_cs224n_gpt/` directory which contains the default final project: Build GPT-2.

## Environment Setup

### Primary Project (GPT-2)
```bash
cd public_cs224n_gpt/
./setup.sh
conda activate cs224n_dfp
```

### Assignment 1 (Word Vectors)
```bash
cd a1/
conda env create -f env.yml
conda activate cs224n
python -m ipykernel install --user --name cs224n
jupyter notebook exploring_word_vectors.ipynb
```

### Assignments 3/4 (Neural Machine Translation)
```bash
cd a3_train/  # or a3/ a4/ a4_train/
pip install -r requirements.txt
```

## Core Commands

### GPT-2 Project (`public_cs224n_gpt/`)

**Testing and Validation:**
```bash
# Test optimizer implementation
python optimizer_test.py

# Sanity check GPT models
python sanity_check.py

# Test classifier
python classifier.py

# Test LoRA implementation
python test_official_lora.py
```

**Training and Inference:**
```bash
# Paraphrase detection (standard)
python paraphrase_detection.py

# Paraphrase detection with LoRA
python paraphrase_detection_lora.py

# Sonnet generation (standard) 
python sonnet_generation.py

# Sonnet generation with LoRA
python sonnet_generation_lora.py
```

**Evaluation:**
```bash
python evaluation.py
```

### Assignment Execution
Most assignments use `run.py`:
```bash
cd a3_train/  # or relevant assignment directory
python run.py [arguments]
```

## Code Architecture

### GPT-2 Implementation Structure

**Core Models** (`models/`):
- `base_gpt.py`: Base GPT model class
- `gpt2.py`: GPT-2 model implementation with missing code blocks to implement

**Core Modules** (`modules/`):
- `attention.py`: Attention mechanisms (contains missing code blocks)
- `gpt2_layer.py`: GPT-2 transformer layers (contains missing code blocks)

**Main Components**:
- `config.py`: Model configuration classes (PretrainedConfig, GPT2Config)
- `classifier.py`: Sentiment classification using GPT models (contains missing code blocks)
- `optimizer.py`: AdamW optimizer implementation (contains missing code blocks) 
- `datasets.py`: Data loading utilities
- `utils.py`: Utility functions and helpers
- `evaluation.py`: Model evaluation utilities

**Task-Specific Scripts**:
- `paraphrase_detection.py`: Cloze-style paraphrase detection
- `sonnet_generation.py`: Autoregressive sonnet generation
- `*_lora.py`: LoRA (Low-Rank Adaptation) versions for fine-tuning

### Key Dependencies
- PyTorch (torch, torchvision, torchaudio)
- transformers
- einops (for tensor operations)
- tokenizers
- sacrebleu (for evaluation)
- sklearn, tqdm, requests

### Important Implementation Notes

**Part 1 Focus Areas** (missing code blocks to implement):
- `modules/attention.py`: Attention mechanisms
- `modules/gpt2_layer.py`: Transformer layers  
- `models/gpt2.py`: Main GPT-2 model
- `classifier.py`: Classification head
- `optimizer.py`: AdamW optimizer

**Part 2 Extensions**: 
- Implement improvements for paraphrase detection task
- Implement improvements for sonnet generation task
- Consider LoRA fine-tuning approach for better task performance

**GPU Memory Management**:
- Adjust batch sizes according to available GPU memory
- Use gradient checkpointing if needed for large models
- Monitor CUDA memory usage during training

### Development Workflow
1. Implement missing code blocks in Part 1 modules
2. Test implementations using provided test scripts
3. Implement and test extensions for Part 2 tasks  
4. Experiment with hyperparameters and model improvements
5. Use LoRA versions for efficient fine-tuning when applicable

### Assignment Structure
- `a1/`: Word vectors exploration (Jupyter notebook)
- `a2/`: Word2Vec implementation  
- `a3/`, `a3_train/`: Neural machine translation
- `a4/`, `a4_train/`: Character-level language modeling with minGPT
- `public_cs224n_gpt/`: Main GPT-2 implementation project