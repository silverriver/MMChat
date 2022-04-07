from collections import defaultdict
import math

def count_ngram(hyps_resp, n):
    """
    Count the number of unique n-grams
    :param hyps_resp: list, a list of responses
    :param n: int, n-gram
    :return: the dict of each ngram count, total frequency
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_entropy get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_entropy takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    ngram_dict = defaultdict(int)
    total_freq = 0.0
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram = ' '.join(resp[i: i + n])
            ngram_dict[ngram] += 1
            total_freq += 1
    return ngram_dict, total_freq

def eval_entropy(hyps_resp, n):
    """
    compute distinct score for the hyps_resp
    :param eval_entropy: list, a list of hyps responses
    :return: entropy-n score
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_entropy get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_entropy takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    hyps_resp = [[str(x) for x in l] for l in hyps_resp]
    hyps_resp = [(' '.join(i)).split() for i in hyps_resp]
    ngram_dict, total_freq = count_ngram(hyps_resp, n)
    _sum = 0.0
    for ngram in ngram_dict:
        log_prob = math.log(ngram_dict[ngram] / total_freq)
        _sum += ngram_dict[ngram] * log_prob
    entropy = - (1/total_freq) * _sum

    return entropy