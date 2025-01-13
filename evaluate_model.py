import json
from sklearn.metrics import precision_score, recall_score, f1_score

# Load test data and predictions with UTF-8 encoding
with open("data\\test_processed.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

with open("model_output\\predictions.json", "r", encoding="utf-8") as f:
    predictions = json.load(f)

# Ensure the lengths match
assert len(test_data) == len(predictions), "Mismatch in test data and predictions length!"

# Helper function to process and split entities
def parse_entities(output):
    return [entity.strip() for entity in output.split(",")]

# Track evaluation metrics
total_true_positives = 0
total_false_positives = 0
total_false_negatives = 0

# Prepare output content
output_lines = []

for i, (test_entry, prediction) in enumerate(zip(test_data, predictions)):
    # Parse ground truth and prediction into sets of entities
    ground_truth = set(parse_entities(test_entry["output"]))
    predicted = set(parse_entities(prediction))
    
    # Calculate true positives, false positives, and false negatives
    true_positives = ground_truth & predicted
    false_positives = predicted - ground_truth
    false_negatives = ground_truth - predicted

    # Aggregate for global metrics
    total_true_positives += len(true_positives)
    total_false_positives += len(false_positives)
    total_false_negatives += len(false_negatives)

    # Prepare sample-wise evaluation details
    output_lines.append(f"Sample {i + 1}:")
    output_lines.append(f"Input: {test_entry['input']}")
    output_lines.append(f"Ground Truth: {ground_truth}")
    output_lines.append(f"Prediction: {predicted}")
    output_lines.append(f"True Positives: {true_positives}")
    output_lines.append(f"False Positives: {false_positives}")
    output_lines.append(f"False Negatives: {false_negatives}")
    output_lines.append("")

# Calculate precision, recall, and F1 score globally
precision = total_true_positives / (total_true_positives + total_false_positives) if total_true_positives + total_false_positives > 0 else 0
recall = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

# Append global metrics to output
output_lines.append("Global Metrics:")
output_lines.append(f"Precision: {precision:.4f}")
output_lines.append(f"Recall: {recall:.4f}")
output_lines.append(f"F1-Score: {f1:.4f}")

# Save output to a file
with open("model_output\\evaluation_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Evaluation results have been saved to 'evaluation_output.txt'.")
