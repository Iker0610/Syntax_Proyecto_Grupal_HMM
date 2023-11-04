import numpy as np
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def accuracy(y_true, y_pred):
    tp_tn = [g_tag for g_tag, p_tag in zip(y_true, y_pred) if g_tag == p_tag]
    return len(tp_tn) / len(y_true)


def precision(y_true, y_pred, tag):
    tp = [g_tag for g_tag, p_tag in zip(y_true, y_pred) if p_tag == g_tag and (p_tag == tag or tag is None)]
    tp_fp = [g_tag for g_tag, p_tag in zip(y_true, y_pred) if p_tag == tag or tag is None]
    return len(tp) / len(tp_fp)


def recall(y_true, y_pred, tag):
    tp = [g_tag for g_tag, p_tag in zip(y_true, y_pred) if p_tag == g_tag and (p_tag == tag or tag is None)]
    tp_fn = [g_tag for g_tag, p_tag in zip(y_true, y_pred) if g_tag == tag or tag is None]
    return len(tp) / len(tp_fn)


def f1(y_true, y_pred, tag):
    p = precision(y_true, y_pred, tag)
    r = recall(y_true, y_pred, tag)
    return 2 * p * r / (p + r)


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
    return np.exp(-np.sum(sentence_log_probabilities) / len(sentence_log_probabilities))


def evaluate_dataset(gold: list[list[tuple[str, str]]], prediction: list[list[tuple[str, str]]], sentence_log_probabilities: list[float]) -> dict[str, Any]:
    # Flatten the lists
    gold = [tag for sentence in gold for tag in sentence]
    prediction = [tag for sentence in prediction for tag in sentence]

    # Get the tags
    tags = list(set([tag for tag, _ in gold]))

    # Calculate the metrics
    acc = accuracy(gold, prediction)
    perp = perplexity(sentence_log_probabilities)
    precisions = {tag: precision(gold, prediction, tag) for tag in tags}
    recalls = {tag: recall(gold, prediction, tag) for tag in tags}
    f1s = {tag: f1(gold, prediction, tag) for tag in tags}

    # Create the confusion matrix
    conf_matrix(gold, prediction, tags)

    return {
        "accuracy": acc,
        "perplexity": perp,
        "per_tag": {tag: {"precision": precisions[tag], "recall": recalls[tag], "f1": f1s[tag]} for tag in tags},
        "macro": {"precision": np.mean(list(precisions.values())), "recall": np.mean(list(recalls.values())), "f1": np.mean(list(f1s.values()))},
        "micro": {"precision": precision(gold, prediction, None), "recall": recall(gold, prediction, None), "f1": f1(gold, prediction, None)}
    }
