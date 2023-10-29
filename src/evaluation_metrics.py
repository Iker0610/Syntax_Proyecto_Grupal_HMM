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
    matrix =  confusion_matrix(y_true, y_pred, labels=labels)

