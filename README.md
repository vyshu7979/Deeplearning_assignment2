Seq2Seq Transliteration Model for Language Conversion
This project implements a Seq2Seq model for the transliteration of text, converting words from one script (Latin) to another (Native Language). The model is trained using LSTM (Long Short-Term Memory) or other RNN (Recurrent Neural Network) variants and is evaluated using a character-level accuracy metric.

Table of Contents
Requirements

Dataset

Setup

Training

Prediction

Evaluation

Results

License

Requirements
Python 3.x

PyTorch

Pandas

NumPy

scikit-learn

Levenshtein

Google Colab (optional)

Dataset
This model requires a dataset in a TSV (Tab-Separated Value) format for training, where each line consists of:

Source (Latin text): The input text in the Latin script.

Target (Native text): The output text in the Native script.

You can upload your own TSV files or use any available transliteration dataset. The dataset is expected to be in the following format:

arduino
Copy
Edit
<Latin text>  <Native script text>
Example:

nginx
Copy
Edit
hello    హలో
world    ప్రపంచం
Setup
To use the provided script, ensure that the necessary libraries are installed. You can install them using pip:

bash
Copy
Edit
pip install torch pandas numpy scikit-learn python-Levenshtein
You can also run this code directly on Google Colab if you want to use Google Drive for storage or upload files manually.

Training
1. Mount Google Drive
You can mount Google Drive to access and upload the training dataset from your local environment or Google Drive.

python
Copy
Edit
drive.mount('/content/drive')
2. Upload the Training and Test Files
Upload the training and testing TSV files when prompted by the script. The script expects files in the following format:

train_file.tsv: Contains training pairs in Latin → Native script format.

test_file.tsv: Contains testing pairs in Latin → Native script format.

3. Set the Configurations
The script defines default configurations for the model, such as:

Embedding size: 128

Hidden size: 256

Batch size: 32

Epochs: 10

Learning rate: 0.001

Teacher forcing ratio: 0.5

These values can be adjusted in the Config class.

4. Train the Model
Once the dataset is uploaded and the configurations are set, you can start training the model. It uses the Seq2Seq architecture, where:

The Encoder processes the input sequence (Latin text).

The Decoder generates the output sequence (Native text).

bash
Copy
Edit
python seq2seq_transliteration.py
Training will output loss values for each epoch.

Prediction
Once the model is trained, you can use it to make predictions (transliterations) from Latin to Native script. Here's a sample usage:

python
Copy
Edit
word = "hello"
transliterated_word = predict(word, max_len=30)
print(f"Transliterated word: {transliterated_word}")
The predict() function takes a Latin word as input and generates the corresponding transliterated output.

Evaluation
The script evaluates the model using Character-Level Accuracy based on the Levenshtein distance between the predicted and actual transliterated text.

The evaluation function computes the character-level accuracy as follows:

python
Copy
Edit
accuracy = char_accuracy(y_true, y_pred)
Where y_true is the list of actual transliterated words (native text), and y_pred is the list of predicted transliterated words.

Results
After training, the script prints the character-level accuracy of the model on the test dataset.

Example Output:
vbnet
Copy
Edit
Epoch 1/10 | Loss: 3.4567
Epoch 2/10 | Loss: 2.9876
...
Char-Level Accuracy on Test Set: 85.34%
License
This code is available for academic and personal use. You may modify it for non-commercial purposes.







