import os
import json
import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    get_scheduler
)
from datasets import Dataset
import numpy as np
from collections import Counter

class TLQATrainer:
    def __init__(self, model_name="facebook/bart-base", output_dir="model_output"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_data(self, train_path, val_path=None):
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        val_data = None
        if val_path:
            with open(val_path, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
        
        return train_data, val_data

    def add_entities_to_tokenizer(self, train_data):
        """Add special entities to the tokenizer's vocabulary."""
        entities = set()
        for item in train_data:
            entities.update(self.extract_entities(item['input']))
            entities.update(self.extract_entities(item['output']))
        
        print(f"Adding {len(entities)} entities to tokenizer vocabulary...")
        self.tokenizer.add_tokens(list(entities))
        self.model.resize_token_embeddings(len(self.tokenizer))

    @staticmethod
    def extract_entities(text):
        """Extract entities wrapped in <ENTITY> tags."""
        return [word.strip() for word in text.split() if word.startswith('<') and word.endswith('>')]

    def preprocess_data(self, examples, max_length=256):
        """Preprocess data with special handling for entities."""
        model_inputs = self.tokenizer(
            examples['input'],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            examples['output'],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Replace padding tokens in labels with -100 to ignore during loss computation
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["labels"][labels["input_ids"] == self.tokenizer.pad_token_id] = -100
        return model_inputs

    def balance_entity_frequencies(self, train_data, min_examples=10):
        """Duplicate underrepresented entities to ensure balance in the dataset."""
        entity_counts = Counter()
        for item in train_data:
            entities = self.extract_entities(item['output'])
            entity_counts.update(entities)

        augmented_data = train_data.copy()
        for entity, count in entity_counts.items():
            if count < min_examples:
                examples = [item for item in train_data if entity in item['output']]
                # Duplicate examples for rare entities
                augmented_data.extend(examples * (min_examples - count))
        
        print(f"Original training examples: {len(train_data)}, Augmented: {len(augmented_data)}")
        return augmented_data

    def train(self, train_data, val_data=None, epochs=3, batch_size=8):
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)

        # Add entities to tokenizer
        self.add_entities_to_tokenizer(train_data)

        # Balance rare entity frequencies
        train_data = self.balance_entity_frequencies(train_data)

        train_dataset = Dataset.from_list([{'input': x['input'], 'output': x['output']} 
                                           for x in train_data])
        
        train_dataset = train_dataset.map(
            self.preprocess_data,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        val_dataset = None
        if val_data:
            val_dataset = Dataset.from_list([{'input': x['input'], 'output': x['output']} 
                                             for x in val_data])
            val_dataset = val_dataset.map(
                self.preprocess_data,
                batched=True,
                remove_columns=val_dataset.column_names
            )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size, 
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(self.output_dir, "logs"),
            save_strategy="epoch",
            eval_strategy="epoch" if val_dataset else "no",
            load_best_model_at_end=True if val_dataset else False,
            learning_rate=3e-5,
            gradient_accumulation_steps=4, 
            fp16=True
        )

        # Create the optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=training_args.learning_rate)

        # Create the scheduler
        num_training_steps = len(train_dataset) * epochs
        scheduler = get_scheduler(
            "linear",  # Can also use 'cosine', 'constant', etc.
            optimizer=optimizer,
            num_warmup_steps=100,  # Warmup steps
            num_training_steps=num_training_steps
        )

        # Initialize Trainer with the optimizer and scheduler
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model),
            optimizers=(optimizer, scheduler)  # Pass optimizer and scheduler here
        )

        # Start training
        self.trainer.train()
        self.trainer.save_model(os.path.join(self.output_dir, "final_model"))

    def predict(self, test_data):
        predictions = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        for item in test_data:
            inputs = self.tokenizer(item['input'], 
                                  return_tensors="pt", 
                                  max_length=256, 
                                  truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(decoded_output)

        return predictions


if __name__ == "__main__":
    TRAIN_PATH = "data/train_processed.json"
    VAL_PATH = "data/val_processed.json"    
    TEST_PATH = "data/test_processed.json"  
    OUTPUT_DIR = "model_output"

    tlqa_trainer = TLQATrainer(model_name="facebook/bart-base", output_dir=OUTPUT_DIR)

    try:
        print("Loading data...")
        train_data, val_data = tlqa_trainer.load_data(TRAIN_PATH, VAL_PATH)
        
        print("Starting training...")
        tlqa_trainer.train(train_data, val_data, epochs=5, batch_size=16)
        
        print("Running inference on test set...")
        with open(TEST_PATH, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        predictions = tlqa_trainer.predict(test_data)
        
        output_file = os.path.join(OUTPUT_DIR, "predictions.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
            
        print(f"Predictions saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
