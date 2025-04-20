# 🔤 Character-Level Seq2Seq Transliteration Model

This project implements a sequence-to-sequence (Seq2Seq) model for transliteration between Latin and native scripts using PyTorch. It includes a custom dataset loader, character-level vocab generation, and training pipeline with character-level accuracy evaluation.

---

## 📚 Dataset

- Format: Tab-separated `.tsv` file with each row containing:
  - Column 0: Native script (target)
  - Column 1: Latin script (source)
- Upload both training and testing `.tsv` files when prompted.
- Vocabulary is built character-wise from training data.

---

## 🏗️ Model Architecture

- **Encoder**
  - Embedding layer
  - RNN (LSTM or GRU)
- **Decoder**
  - Embedding layer
  - RNN (LSTM or GRU)
  - Linear layer to project to output vocab
- **Seq2Seq Wrapper**
  - Uses teacher forcing during training
  - Generates output character-by-character

---

## ⚙️ Training Configuration

| Parameter               | Value     |
|------------------------|-----------|
| Embedding Dimension     | 128       |
| Hidden Dimension        | 256       |
| RNN Layers              | 1         |
| Cell Type               | LSTM      |
| Batch Size              | 32        |
| Epochs                  | 10        |
| Learning Rate           | 0.001     |
| Teacher Forcing Ratio   | 0.5       |
| Padding Token Index     | 0         |

---

## 🚀 Training Steps

1. Mount Google Drive and upload dataset files.
2. Preprocess and build input/output vocabularies.
3. Create a custom PyTorch `Dataset` and `DataLoader`.
4. Train the Seq2Seq model using teacher forcing.
5. Evaluate on test set using character-level accuracy.

---

## 📈 Evaluation Metrics

### 🔡 Character-Level Accuracy

- Metric:  
  `Accuracy = 1 - (Edit Distance / Total Characters)`
- Uses Levenshtein distance between predicted and true strings.

---

## 🧪 Sample Output

```bash
🔤 Sample Test Predictions:
namaste → नमस्ते
shukriya → शुक्रिया
krishna → कृष्ण

✅ Char-Level Accuracy on Test Set: 87.35%
```

#  GPT-2  Finetune the GPT2 model to generate lyrics for English songs.

This `README.md` file explains the dataset, model architecture, training steps, evaluation metrics, and usage instructions:
1. **Fine-tune GPT-2 to generate English song lyrics**


---

## 📁 Project 1: Fine-tune GPT-2 for English Song Lyrics

### 📚 Dataset

- Source: [Kaggle - paultimothymooney/poetry](https://www.kaggle.com/datasets/paultimothymooney/poetry)
- Contains a collection of poetry texts.
- The `.txt` files are aggregated into a single `lyrics.txt` file for training.

### 🏗️ Model Architecture

- Pretrained model: `gpt2` from HuggingFace Transformers.
- Language modeling head on top of GPT-2 base.
- Tokenizer: `GPT2Tokenizer`.

### 🧪 Training Setup

- **Block size:** 128
- **Epochs:** 3
- **Batch size:** 2
- **Learning Rate:** Default from `Trainer`
- **Loss function:** Cross-entropy (via `Trainer`)
- **Data Collator:** `DataCollatorForLanguageModeling` (causal language modeling)

### 🚀 Training Steps

1. Aggregate poetry data into `lyrics.txt`.
2. Tokenize using GPT-2 tokenizer.
3. Create a `TextDataset` for training.
4. Set up a `Trainer` object with GPT-2 and begin training.

### 📈 Evaluation Metrics

- No validation split in this simple setup.
- Evaluation based on qualitative output (generated lyrics).
- Sample lyrics generated from a prompt like `"In the moonlight I dream"`.

### 💻 Usage Example

```python
prompt = "In the moonlight I dream"
print(generate_lyrics(prompt, max_length=50))
🎤 Prompt: In the moonlight I dream
🎵 Generated Lyrics: 
In the moonlight I dream  
Of stars that softly gleam  
A whispered song flows through the night  
With silver winds and glowing light...

