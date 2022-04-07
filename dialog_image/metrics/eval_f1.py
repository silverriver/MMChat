def eval_f1(ref, pred):
    """
    :param ref: list(list(list(any))), a list of reference sentences, each element of the list is a list of references
    :param pred: list(list(any)), a list of predictions
    :return: f1 score
    """
    assert len(ref) == len(pred) > 0
    precisions = []
    recalls = []
    for i, s in enumerate(pred):
        ref_set = set()
        for rs in ref[i]:
            for w in rs:
                ref_set.add(w)
        pred_set = set()
        for w in s:
            pred_set.add(w)

        p = 0
        for w in s:
            if w in ref_set:
                p += 1
        if len(s) > 0:
            p /= len(s)
        r = 0
        for rs in ref[i]:
            for w in rs:
                if w in pred_set:
                    r += 1
        tot_l = sum([len(rs) for rs in ref[i]])
        if tot_l > 0:
            r /= tot_l

        precisions.append(p)
        recalls.append(r)

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    return 0.0 if precision == recall == 0 else 2 * precision * recall / (precision + recall)


if __name__ == '__main__':
    print(eval_f1([[['rabbit', 'carrot'], ['rabbit', 'carrot']],
                   [['yes', 'no', 'no'], ['yes', 'fxxk']]], [['rabbit', 'carrot'], ['yes', 'no', 'no']]))