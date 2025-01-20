import json

# Load test data and predictions
with open("data\\test_processed.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

with open("model_output\\predictions.json", "r", encoding="utf-8") as f:
    predictions = json.load(f)

# Ensure lengths match
assert len(test_data) == len(predictions), "Mismatch in test data and predictions length!"

# Helper function to parse entities and timelines
def parse_entities_and_timelines(output):
    entities = []
    for item in output.split(","):
        item = item.strip()
        if "(" in item and ")" in item:
            entity, timeline = item.rsplit("(", 1)
            entities.append((entity.strip(), timeline.strip(")")))
        else:
            entities.append((item.strip(), None))
    return entities

# Initialize metrics
total_true_positives = 0
total_false_positives = 0
total_false_negatives = 0
total_timeline_matches = 0
total_timeline_mismatches = 0
total_ground_truth_entities = 0

# Prepare output lines
output_lines = []

for i, (test_entry, prediction) in enumerate(zip(test_data, predictions)):
    # Parse ground truth and prediction
    ground_truth = set(parse_entities_and_timelines(test_entry["output"]))
    predicted = set(parse_entities_and_timelines(prediction))

    # Extract entities and timelines separately
    ground_truth_entities = {entity for entity, _ in ground_truth}
    predicted_entities = {entity for entity, _ in predicted}

    # Calculate matches
    true_positives = ground_truth_entities & predicted_entities
    false_positives = predicted_entities - ground_truth_entities
    false_negatives = ground_truth_entities - predicted_entities

    # Evaluate timelines for matched entities
    timeline_matches = 0
    timeline_mismatches = 0
    for entity, timeline in ground_truth:
        if entity in predicted_entities:
            predicted_timeline = next((t for e, t in predicted if e == entity), None)
            if timeline == predicted_timeline:
                timeline_matches += 1
            else:
                timeline_mismatches += 1

    # Aggregate metrics
    total_true_positives += len(true_positives)
    total_false_positives += len(false_positives)
    total_false_negatives += len(false_negatives)
    total_timeline_matches += timeline_matches
    total_timeline_mismatches += timeline_mismatches
    total_ground_truth_entities += len(ground_truth_entities)

    # Prepare sample-wise evaluation details
    output_lines.append(f"Sample {i + 1}:")
    output_lines.append(f"Input: {test_entry['input']}")
    output_lines.append(f"Ground Truth: {ground_truth}")
    output_lines.append(f"Prediction: {predicted}")
    output_lines.append(f"True Positives (Entities): {true_positives}")
    output_lines.append(f"False Positives (Entities): {false_positives}")
    output_lines.append(f"False Negatives (Entities): {false_negatives}")
    output_lines.append(f"Timeline Matches: {timeline_matches}")
    output_lines.append(f"Timeline Mismatches: {timeline_mismatches}")
    output_lines.append("")

# Calculate global metrics
precision = total_true_positives / (total_true_positives + total_false_positives) if total_true_positives + total_false_positives > 0 else 0
recall = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

timeline_accuracy = total_timeline_matches / (total_timeline_matches + total_timeline_mismatches) if total_timeline_matches + total_timeline_mismatches > 0 else 0
completeness = total_true_positives / total_ground_truth_entities if total_ground_truth_entities > 0 else 0

# Append global metrics to output
output_lines.append("Global Metrics:")
output_lines.append(f"Precision (Entities): {precision:.4f}")
output_lines.append(f"Recall (Entities): {recall:.4f}")
output_lines.append(f"F1-Score (Entities): {f1:.4f}")
output_lines.append(f"Timeline Accuracy: {timeline_accuracy:.4f}")
output_lines.append(f"Completeness: {completeness:.4f}")

# Save output to a file
with open("model_output\\evaluation_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Evaluation results have been saved to 'evaluation_output.txt'.")
