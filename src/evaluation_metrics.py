def accuracy(y_true, y_pred):
    check = [g_tag for g_tag, p_tag in zip(y_true, y_pred) if g_tag == p_tag]
    return len(check) / len(y_true)


def precision(y_true, y_pred):
    # TODO
    return 0.0


def recall(y_true, y_pred):
    # TODO
    return 0.0


def f1(y_true, y_pred):
    # TODO
    return 0.0
