# Project Completion Checklist

## Data and Environment

- [x] Confirm the 21 artist CSV files and required columns
- [x] Configure the Python 3.11 CUDA environment
- [x] Load, clean, deduplicate, and validate lyrics
- [x] Create stratified train, validation, and test splits
- [x] Save aggregate dataset statistics

## Model Development

- [x] Download the `openai-community/gpt2` baseline
- [x] Complete a GPU smoke test
- [x] Fine-tune GPT-2 for three epochs
- [x] Save checkpoints and the final model locally
- [x] Record training and validation metrics

## Generation and Evaluation

- [x] Generate baseline GPT-2 samples
- [x] Generate fine-tuned samples with temperature, top-k, and top-p decoding
- [x] Build an interactive title-based comparison script
- [x] Calculate baseline and fine-tuned test perplexity
- [x] Complete an AI-assisted composer-style qualitative comparison

## Analysis and Sharing

- [x] Create an executed results notebook
- [x] Explain loss, perplexity, overfitting, and decoding strategies
- [x] Discuss results, limitations, and recommended decoding method
- [x] Prepare a curated public GitHub portfolio repository
- [ ] Replace or supplement AI-assisted scores with independent human ratings
- [ ] Complete the group report
