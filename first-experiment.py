# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

from KnnSearch import KnnSearch

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# Make predictions
def generate_predictions(model, tokenizer, inputs):
    inputs_tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs_tokenized["input_ids"], max_length=128)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return predictions

# Prepare few-shot examples
def prepare_few_shot_examples(dataset):
    examples = []
    for sample in dataset:
        examples.append(f"Q: {sample['input']} A: {sample['output']}")
    return "\n".join(examples)

# Format input for few-shot prediction
def format_input(few_shot_examples, test_question):
    return f"{few_shot_examples}\nQ: {test_question} A:"

# Load TLQA dataset
with open("train_processed.json", "r") as file:
    train_dataset = json.load(file)
with open("test_processed.json", "r") as file:
    test_dataset = json.load(file)

knn_instance = KnnSearch()
transfer_questions = knn_instance.get_transfer_questions(train_dataset)
data_emb = knn_instance.get_embeddings_for_data(transfer_questions)

# Predictions on test set
for test_sample in test_dataset[:2]:
    test_question = test_sample["input"]

    few_shot_examples = knn_instance.get_top_n_neighbours(test_question, data_emb, train_dataset,3)
    few_shot_examples_prompt = prepare_few_shot_examples(few_shot_examples)

    # Format input for few-shot prediction
    input_text = format_input(few_shot_examples_prompt, test_question)

    print("\nInput text:", input_text)

    # Predict with both models
    print("\nTest Question:", test_question)

    print("Predictions using Flan-T5-Large:")
    predictions_large = generate_predictions(model, tokenizer, [input_text])
    print(predictions_large)
    print("Actual answer: ", test_sample["output"])

