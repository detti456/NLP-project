import json
import numpy as np
from evaluate import load

# Helper function for BLEU and ROUGE metrics
class TLQAMetrics:
    def evaluate_predictions(self, predictions, references):
        """Evaluate BLEU and ROUGE scores."""
        # Load metrics
        bleu = load('bleu')
        rouge = load('rouge')

        # Compute BLEU and ROUGE
        bleu_scores = bleu.compute(predictions=predictions, references=references)
        rouge_scores = rouge.compute(predictions=predictions, references=references)

        return {
            "BLEU": bleu_scores,
            "ROUGE": rouge_scores,
        }

# Initialize BLEU/ROUGE evaluator
metrics = TLQAMetrics()

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

# Initialize global (micro) metrics
total_true_positives = 0
total_false_positives = 0
total_false_negatives = 0
total_timeline_matches = 0
total_timeline_mismatches = 0
total_ground_truth_entities = 0

# Lists to store sample-wise (macro) metrics
sample_precisions = []
sample_recalls = []
sample_f1s = []
timeline_matches = []
timeline_mismatches = []

# Lists for BLEU and ROUGE references and predictions
references = []
sample_predictions = []

# Prepare output lines for sample-wise and global results
macro_output_lines = []
micro_output_lines = []

for i, (test_entry, prediction) in enumerate(zip(test_data, predictions)):
    # Parse ground truth and prediction
    ground_truth = set(parse_entities_and_timelines(test_entry["output"]))
    predicted = set(parse_entities_and_timelines(prediction))

    # Extract entities and timelines separately
    ground_truth_entities = {entity for entity, _ in ground_truth}
    predicted_entities = {entity for entity, _ in predicted}

    # Add reference and prediction for BLEU/ROUGE
    references.append(test_entry["output"])
    sample_predictions.append(prediction)

    # Calculate matches
    true_positives = ground_truth_entities & predicted_entities
    false_positives = predicted_entities - ground_truth_entities
    false_negatives = ground_truth_entities - predicted_entities

    # Precision, recall, F1 for the current sample
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if len(true_positives) + len(false_positives) > 0 else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if len(true_positives) + len(false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Append sample-wise metrics
    sample_precisions.append(precision)
    sample_recalls.append(recall)
    sample_f1s.append(f1)

    # Evaluate timelines for matched entities
    sample_timeline_matches = 0
    sample_timeline_mismatches = 0
    for entity, timeline in ground_truth:
        if entity in predicted_entities:
            predicted_timeline = next((t for e, t in predicted if e == entity), None)
            if timeline == predicted_timeline:
                sample_timeline_matches += 1
            else:
                sample_timeline_mismatches += 1

    timeline_matches.append(sample_timeline_matches)
    timeline_mismatches.append(sample_timeline_mismatches)

    # Aggregate metrics for micro-averaging
    total_true_positives += len(true_positives)
    total_false_positives += len(false_positives)
    total_false_negatives += len(false_negatives)
    total_timeline_matches += sample_timeline_matches
    total_timeline_mismatches += sample_timeline_mismatches
    total_ground_truth_entities += len(ground_truth_entities)

    # Prepare sample-wise evaluation details for macro output
    macro_output_lines.append(f"Sample {i + 1}:")
    macro_output_lines.append(f"Input: {test_entry['input']}")
    macro_output_lines.append(f"Ground Truth: {ground_truth}")
    macro_output_lines.append(f"Prediction: {predicted}")
    macro_output_lines.append(f"Precision: {precision:.4f}")
    macro_output_lines.append(f"Recall: {recall:.4f}")
    macro_output_lines.append(f"F1-Score: {f1:.4f}")
    macro_output_lines.append(f"Timeline Matches: {sample_timeline_matches}")
    macro_output_lines.append(f"Timeline Mismatches: {sample_timeline_mismatches}")
    macro_output_lines.append("")

# Calculate macro-averaged metrics
macro_precision = np.mean(sample_precisions)
macro_recall = np.mean(sample_recalls)
macro_f1 = np.mean(sample_f1s)

# Calculate overall timeline accuracy for macro results
macro_timeline_accuracy = sum(timeline_matches) / (sum(timeline_matches) + sum(timeline_mismatches)) if sum(timeline_matches) + sum(timeline_mismatches) > 0 else 0

# Append macro-averaged metrics to macro output
macro_output_lines.append("Global Macro Metrics:")
macro_output_lines.append(f"Macro Precision (Entities): {macro_precision:.4f}")
macro_output_lines.append(f"Macro Recall (Entities): {macro_recall:.4f}")
macro_output_lines.append(f"Macro F1-Score (Entities): {macro_f1:.4f}")
macro_output_lines.append(f"Macro Timeline Accuracy: {macro_timeline_accuracy:.4f}")

# Calculate global (micro) metrics
micro_precision = total_true_positives / (total_true_positives + total_false_positives) if total_true_positives + total_false_positives > 0 else 0
micro_recall = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0
micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0

micro_timeline_accuracy = total_timeline_matches / (total_timeline_matches + total_timeline_mismatches) if total_timeline_matches + total_timeline_mismatches > 0 else 0
completeness = total_true_positives / total_ground_truth_entities if total_ground_truth_entities > 0 else 0

# Append global metrics to micro output
micro_output_lines.append("Global Micro Metrics:")
micro_output_lines.append(f"Micro Precision (Entities): {micro_precision:.4f}")
micro_output_lines.append(f"Micro Recall (Entities): {micro_recall:.4f}")
micro_output_lines.append(f"Micro F1-Score (Entities): {micro_f1:.4f}")
micro_output_lines.append(f"Micro Timeline Accuracy: {micro_timeline_accuracy:.4f}")
micro_output_lines.append(f"Completeness: {completeness:.4f}")

# Evaluate BLEU and ROUGE scores
bleu_rouge_results = metrics.evaluate_predictions(sample_predictions, references)

# Append BLEU and ROUGE to macro output
macro_output_lines.append("Global BLEU and ROUGE Metrics:")
macro_output_lines.append(f"BLEU: {bleu_rouge_results['BLEU']}")
macro_output_lines.append(f"ROUGE: {bleu_rouge_results['ROUGE']}")

# Save macro results to a file
with open("model_output\\evaluation_output_macro.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(macro_output_lines))

# Save micro results to a separate file
with open("model_output\\evaluation_output_micro.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(micro_output_lines))

print("Evaluation results saved to 'evaluation_output_macro.txt' and 'evaluation_output_micro.txt'.")
