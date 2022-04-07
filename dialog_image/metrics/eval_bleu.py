from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction


smooth = SmoothingFunction()


def eval_bleu(ref, pred):
    """
    :param ref: list(list(list(any))), a list of reference sentences, each element of the list is a list of references
    :param pred: list(list(any)), a list of predictions
    :return: corpus bleu score
    """
    return corpus_bleu(ref, pred, smoothing_function=smooth.method1) * 100


def eval_bleu_detail(ref, pred):
    """
    :param ref: list(list(list(any))), a list of reference sentences, each element of the list is a list of references
    :param pred: list(list(any)), a list of predictions
    :return: corpus bleu score
    """
    return [f'{corpus_bleu(ref, pred, weights=[1, 0, 0, 0], smoothing_function=smooth.method1) *100:.3f}',
           f'{corpus_bleu(ref, pred, weights=[0.5, 0.5, 0, 0], smoothing_function=smooth.method1) *100:.3f}',
           f'{corpus_bleu(ref, pred, weights=[0.333, 0.333, 0.333, 0], smoothing_function=smooth.method1) *100:.3f}', 
           f'{corpus_bleu(ref, pred, weights=[0.25, 0.25, 0.25, 0.25], smoothing_function=smooth.method1) *100:.3f}']
