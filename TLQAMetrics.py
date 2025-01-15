import re
import string
from statistics import mean
import collections
from evaluate import load

class TLQAMetrics:
    def normalize_answer(self, s):
      """Lower text and remove punctuation, articles and extra whitespace."""
      def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
      def white_space_fix(text):
        return ' '.join(text.split())
      def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
      def lower(text):
        return text.lower()
      return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(self, s):
      if not s: return []
      return self.normalize_answer(s).split()

    def compute_exact(self, a_gold, a_pred):
      return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))

    def compute_f1(self, a_gold, a_pred):
      gold_toks = self.get_tokens(a_gold)
      pred_toks = self.get_tokens(a_pred)
      common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
      num_same = sum(common.values())
      if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
      if num_same == 0:
        return 0
      precision = 1.0 * num_same / len(pred_toks)
      recall = 1.0 * num_same / len(gold_toks)
      f1 = (2 * precision * recall) / (precision + recall)
      return f1

    # Example: Evaluate predictions against ground truth
    def evaluate_predictions(self, predictions, references):
        # Load evaluation metrics
        bleu = load('bleu')
        rouge = load('rouge')

        # ROUGE Scores
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        bleu_scores = bleu.compute(predictions=predictions, references=references)

        f1_scores = [self.compute_f1(ref,pred) for ref,pred in zip(references, predictions)]

        avg_f1_score = mean(f1_scores)

        return {
            "F1 Scores": f1_scores,
            "Average F1 Score": avg_f1_score,
            "BLEU": bleu_scores,
            "ROUGE": rouge_scores,
        }