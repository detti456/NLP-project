import pickle


from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from TLQAMetrics import TLQAMetrics
from KnnSearch import KnnSearch

with open("test_processed.json", "r") as file:
    test_dataset = json.load(file)
actual_answers = [sample["output"] for sample in test_dataset[:2]]

ks = [3,5,7,10]


# Make predictions
def generate_predictions(model, tokenizer, inputs):
    inputs_tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs_tokenized["input_ids"], max_length=128)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return predictions


# Prepare few-shot examples
def prepare_few_shot_examples(dataset):
    examples = []  # "Given the following examples, give an answer for the last question.\n"
    for sample in dataset:
        examples.append(f"Q: {sample['input']} A: {sample['output']}")
    return "\n".join(examples)


# Format input for few-shot prediction
def format_input(few_shot_examples, test_question):
    return f"{few_shot_examples}\nQ: {test_question} A:"

for k in ks:
    try:
        with open(f"predictions/flan-t5-large-k{k}-predictions.json", "r") as file:
            predictions = json.load(file)

    except:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

        # Load TLQA dataset
        with open("train_processed.json", "r") as file:
            train_dataset = json.load(file)


        knn_instance = KnnSearch()

        try:
            with open("train_embeddings", 'rb') as f:
                data_emb = pickle.load(f)
        except:
            transfer_questions = knn_instance.get_transfer_questions(train_dataset)
            data_emb = knn_instance.get_embeddings_for_data(transfer_questions)
            with open("train_embeddings", 'wb') as f:
                pickle.dump(data_emb, f)

        predictions = []

        # Predictions on test set
        for test_sample in test_dataset[:2]:
            test_question = test_sample["input"]

            few_shot_examples = knn_instance.get_top_n_neighbours(test_question, data_emb, train_dataset,k)
            few_shot_examples_prompt = prepare_few_shot_examples(few_shot_examples)

            # Format input for few-shot prediction
            input_text = format_input(few_shot_examples_prompt, test_question)

            # print("\nInput text:", input_text)

            # print("Predictions using Flan-T5-Large:")
            predictions_large = generate_predictions(model, tokenizer, [input_text])
            print("Predicted answer: ",predictions_large[0])
            # print("Actual answer: ", test_sample["output"])

            predictions.append(predictions_large[0])

        actual_answers = [sample["output"] for sample in test_dataset[:2]]

        with open(f"predictions/flan-t5-large-k{k}-predictions.json", "w") as file:
            json.dump(predictions, file)

    evaluator = TLQAMetrics()

    # Evaluate
    evaluation_results = evaluator.evaluate_predictions(predictions, actual_answers)

    print("Evaluation Results:")
    print("F1 Scores:", evaluation_results["F1 Scores"])
    print("Average F1 Score:", evaluation_results["Average F1 Score"])
    print("ROUGE:", evaluation_results["ROUGE"])
    print("BLEU:", evaluation_results["BLEU"])

    with open(f"results/flan-t5-large-k{k}-evaluation_results.json", 'w') as f:
        json.dump(evaluation_results, f)