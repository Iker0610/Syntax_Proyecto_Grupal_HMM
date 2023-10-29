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


def f1(y_true, y_pred):
    # TODO
    return 0.0
