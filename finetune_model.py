from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Load preprocessed dataset
train_dataset = Dataset.from_json("train_processed.json")
val_dataset = Dataset.from_json("val_processed.json")

# Load model and tokenizer
model_name = "google/flan-t5-small"  # or "facebook/bart-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the data with padding and truncation
def preprocess_function(examples):
    inputs = tokenizer(examples["input"], max_length=512, truncation=True, padding="max_length")  # padding to max_length
    labels = tokenizer(examples["output"], max_length=512, truncation=True, padding="max_length")  # padding to max_length
    inputs["labels"] = labels["input_ids"]  # Ensure labels are the tokenized input_ids
    return inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision training
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)
torch.cuda.empty_cache()
trainer.train()
