import json

# Load your dataset
with open("train_TLQA.json", "r") as file:
    data_train = json.load(file)
with open("val_TLQA.json", "r") as file:
    data_val = json.load(file)
with open("test_TLQA.json", "r") as file:
    data_test = json.load(file)

train_data = []
val_data = []
test_data = []

for item in data_train:
    input_text = item["question"]
    output_text = ", ".join(item["final_answers"])
    train_data.append({"input": input_text, "output": output_text})

# Save preprocessed data
with open("train_processed.json", "w") as file:
    json.dump(train_data, file)

for item in data_val:
    input_text = item["question"]
    output_text = ", ".join(item["final_answers"])
    val_data.append({"input": input_text, "output": output_text})

# Save preprocessed data
with open("val_processed.json", "w") as file:
    json.dump(val_data, file)

for item in data_test:
    input_text = item["question"]
    output_text = ", ".join(item["final_answers"])
    test_data.append({"input": input_text, "output": output_text})

# Save preprocessed data
with open("test_processed.json", "w") as file:
    json.dump(test_data, file)