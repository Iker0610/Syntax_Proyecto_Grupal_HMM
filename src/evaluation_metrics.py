from typing import Any

import numpy as np
import plotly.figure_factory as ff
from numpy import ndarray
from sklearn.metrics import confusion_matrix


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    tp_tn = np.sum(np.asarray(y_true) == np.asarray(y_pred))
    return np.nan_to_num(tp_tn / len(y_true))


def true_positives(y_true: ndarray, y_pred: ndarray, tag: str) -> int:
    return np.sum((y_true == y_pred) & (y_pred == tag))


def precision(y_true: list[str], y_pred: list[str], tag: str) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = true_positives(y_true, y_pred, tag)
    tp_fp = np.sum(y_pred == tag)
    return np.nan_to_num(tp / tp_fp)


def recall(y_true, y_pred, tag):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = true_positives(y_true, y_pred, tag)
    tp_fn = np.sum(y_true == tag)
    return np.nan_to_num(tp / tp_fn)


def f1(y_true, y_pred, tag):
    p = precision(y_true, y_pred, tag)
    r = recall(y_true, y_pred, tag)

    return np.nan_to_num(2 * p * r / (p + r))


def micro_precision_recall(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.nan_to_num(np.sum(y_true == y_pred) / len(y_pred))


def micro_f1(y_true, y_pred):
    p = r = micro_precision_recall(y_true, y_pred)
    return np.nan_to_num(2 * p * r / (p + r))


def conf_matrix(y_true, y_pred, labels):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    mat_text = [[str(y) for y in x] for x in matrix]

    fig = ff.create_annotated_heatmap(matrix, x=labels, y=labels, annotation_text=mat_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                      # xaxis = dict(title='x'),
                      # yaxis = dict(title='x')
                      )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14), x=0.5, y=-0.15, showarrow=False, text="Predicted value", xref="paper", yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14), x=-0.35, y=0.5, showarrow=False, text="Real value", textangle=-90, xref="paper", yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()


def perplexity(sentence_log_probabilities: list[float]):
    return np.exp2(-np.sum(sentence_log_probabilities) / len(sentence_log_probabilities))


def evaluate_dataset(gold: list[list[tuple[str, str]]], prediction: list[list[tuple[str, str]]], sentence_log_probabilities: list[float]) -> dict[str, Any]:
    # Flatten the lists
    gold_ = [tag[1] for sentence in gold for tag in sentence]
    pred_ = [tag[1] for sentence in prediction for tag in sentence]

    # --------------------------------------

    # Per sentence evaluation
    per_sentence_metrics = []
    for gold_sentence, pred_sentence, sentence_log_prob in zip(gold, prediction, sentence_log_probabilities):
        gold_sentence = [tag[1] for tag in gold_sentence]
        pred_sentence = [tag[1] for tag in pred_sentence]

        # Get the tags
        tags = list(set(gold_sentence) | set(pred_sentence))

        sentence_precisions = {tag: precision(gold_sentence, pred_sentence, tag) for tag in tags}
        sentence_recalls = {tag: recall(gold_sentence, pred_sentence, tag) for tag in tags}
        sentence_f1s = {tag: f1(gold_sentence, pred_sentence, tag) for tag in tags}

        per_sentence_metrics.append({
            "length": len(gold_sentence),
            "accuracy": accuracy(gold_sentence, pred_sentence),
            "probability": np.exp2(sentence_log_prob),
            "per_tag": {tag: {"precision": precision(gold_sentence, pred_sentence, tag), "recall": recall(gold_sentence, pred_sentence, tag), "f1": f1(gold_sentence, pred_sentence, tag)} for tag in tags},
            "macro": {"precision": np.mean(list(sentence_precisions.values())), "recall": np.mean(list(sentence_recalls.values())), "f1": np.mean(list(sentence_f1s.values()))},
            "micro": {"precision": micro_precision_recall(gold_sentence, pred_sentence), "recall": micro_precision_recall(gold_sentence, pred_sentence), "f1": micro_f1(gold_sentence, pred_sentence)}
        })

    # --------------------------------------
    # Get the tags
    tags = list(set(gold_) | set(pred_))

    # Calculate dataset metrics
    acc = accuracy(gold_, pred_)
    perp = perplexity(sentence_log_probabilities)
    precisions = {tag: precision(gold_, pred_, tag) for tag in tags}
    recalls = {tag: recall(gold_, pred_, tag) for tag in tags}
    f1s = {tag: f1(gold_, pred_, tag) for tag in tags}

    return {
        "accuracy": acc,
        "perplexity": perp,
        "per_tag": {tag: {"precision": precisions[tag], "recall": recalls[tag], "f1": f1s[tag]} for tag in tags},
        "macro": {"precision": np.mean(list(precisions.values())), "recall": np.mean(list(recalls.values())), "f1": np.mean(list(f1s.values()))},
        "micro": {"precision": micro_precision_recall(gold_, pred_), "recall": micro_precision_recall(gold_, pred_), "f1": micro_f1(gold_, pred_)},
        "per_sentence": per_sentence_metrics,
    }
