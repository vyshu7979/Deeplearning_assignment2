import os
os.environ["WANDB_DISABLED"] = "true"

import kagglehub
import pandas as pd
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling
)

print("Downloading dataset from KaggleHub...")
dataset_path = kagglehub.dataset_download("paultimothymooney/poetry")
print("Dataset downloaded!")
print("Path to dataset files:", dataset_path)

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        print("Found file:", os.path.join(root, file))

output_path = "lyrics.txt"
with open(output_path, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as infile:
                    content = infile.read().strip()
                    if content:
                        outfile.write(content + "\n\n")

print(f"All poetry compiled into '{output_path}'")

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=output_path,
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./gpt2-lyrics",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

def generate_lyrics(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Lyrics:")
print(generate_lyrics("In the moonlight I dream", max_length=50))
