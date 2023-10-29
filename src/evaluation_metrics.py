import plotly.figure_factory as ff

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def accuracy(y_true, y_pred):
    check = [g_tag for g_tag, p_tag in zip(y_true, y_pred) if g_tag == p_tag]
    return len(check) / len(y_true)


def precision(y_true, y_pred, tag):
    tp = [g_tag for g_tag, p_tag in zip(y_true, y_pred) if p_tag == p_tag and p_tag == tag]
    tp_fp = [g_tag for g_tag, p_tag in zip(y_true, y_pred) if p_tag == tag]
    return len(tp) / len(tp_fp)


def recall(y_true, y_pred, tag):
    tp = [g_tag for g_tag, p_tag in zip(y_true, y_pred) if p_tag == p_tag and p_tag == tag]
    tp_fn = [g_tag for g_tag, p_tag in zip(y_true, y_pred) if g_tag == tag]
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

