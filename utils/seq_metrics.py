# https://github.com/huggingface/datasets/blob/master/metrics/f1/f1.py
# https://github.com/huggingface/datasets/blob/master/metrics/accuracy/accuracy.py
# https://github.com/huggingface/datasets/blob/master/metrics/matthews_correlation/matthews_correlation.py
# https://github.com/huggingface/datasets/blob/master/metrics/precision/precision.py
# https://github.com/huggingface/datasets/blob/master/metrics/recall/recall.py

from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, precision_score, recall_score
import datasets

class SeqMetrics(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"],
        )

    def _compute(self, predictions, references, labels=None, pos_label=1, normalize=True, average=None, sample_weight=None):
        if average is None:
            methods = ['macro','weighted','binary']
        else:
            methods = [average] # str
        
        results = {
            "n": len(predictions),
            "accuracy": float(accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)),
            "matthews_correlation": float(matthews_corrcoef(references, predictions, sample_weight=sample_weight)),
            }
        for a in methods:
            f_score = f1_score(
                references, predictions, labels=labels, pos_label=pos_label, average=a, sample_weight=sample_weight
            )
            p_score = precision_score(
                references, predictions, labels=labels, pos_label=pos_label, average=a, sample_weight=sample_weight
            )
            r_score = recall_score(
                references, predictions, labels=labels, pos_label=pos_label, average=a, sample_weight=sample_weight
            )

            results[f"{a}_precision"] = float(p_score) if p_score.size == 1 else p_score
            results[f"{a}_recall"] =  float(r_score) if r_score.size == 1 else r_score,
            results[f"{a}_f1"] = float(f_score) if f_score.size == 1 else f_score,

        return results