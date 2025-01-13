import json

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
    input_text = item.get("question", "").strip()  
    output_text = ", ".join([ans for ans in item.get("final_answers", []) if ans.strip()]) 

    if input_text and output_text:
        train_data.append({"input": input_text, "output": output_text})

with open("train_processed.json", "w", encoding="utf-8") as file:
    json.dump(train_data, file, ensure_ascii=False)

for item in data_val:
    input_text = item.get("question", "").strip()
    output_text = ", ".join([ans for ans in item.get("final_answers", []) if ans.strip()])
    
    if input_text and output_text:
        val_data.append({"input": input_text, "output": output_text})

with open("val_processed.json", "w", encoding="utf-8") as file:
    json.dump(val_data, file, ensure_ascii=False)

for item in data_test:
    input_text = item.get("question", "").strip()
    output_text = ", ".join([ans for ans in item.get("final_answers", []) if ans.strip()])
    
    if input_text and output_text:
        test_data.append({"input": input_text, "output": output_text})

with open("test_processed.json", "w", encoding="utf-8") as file:
    json.dump(test_data, file, ensure_ascii=False)
