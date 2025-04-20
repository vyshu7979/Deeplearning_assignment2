# Seq2Seq Transliteration

This project implements a **Seq2Seq (Sequence-to-Sequence)** model for transliterating text. The model is built using **PyTorch** and trained on a custom dataset to convert Latin script to a native language script (or vice versa) using character-level translation.

## Features
- **Seq2Seq Model:** An encoder-decoder architecture for text transliteration.
- **LSTM-based RNN:** Uses LSTM (Long Short-Term Memory) units for both encoder and decoder.
- **Character-level Training:** The model works at the character level and can learn to map characters from one script to another.
- **Teacher Forcing:** Implements teacher forcing during training for improved convergence.
- **Metrics:** The model uses character-level accuracy to evaluate performance on test data.

## Requirements

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- Levenshtein (for computing string distances)
- pandas
- scikit-learn
- Google Colab (for easy file uploading and mounting)

Install necessary packages:
```bash
pip install torch pandas scikit-learn python-Levenshtein

