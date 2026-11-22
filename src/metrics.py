import math
import re
from collections import Counter

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def compute_bleu(hypotheses, references, max_n=4):
    """
    Computes standard cumulative BLEU-1 to BLEU-4 score with brevity penalty.
    """
    smooth = 1e-9
    log_precisions = []
    c_total = 0
    r_total = 0

    for n in range(1, max_n + 1):
        matches = 0
        possible = 0
        for hyp, ref in zip(hypotheses, references):
            h_toks = hyp.strip().split()
            r_toks = ref.strip().split()

            if n == 1:
                c_total += len(h_toks)
                r_total += len(r_toks)

            h_ngrams = Counter(ngrams(h_toks, n))
            r_ngrams = Counter(ngrams(r_toks, n))

            for ng, count in h_ngrams.items():
                matches += min(count, r_ngrams.get(ng, 0))
            possible += max(0, len(h_toks) - n + 1)

        p_n = (matches + smooth) / (possible + smooth)
        log_precisions.append(math.log(p_n))

    if c_total == 0:
        return 0.0

    bp = 1.0 if c_total > r_total else math.exp(1.0 - (r_total / (c_total + 1e-9)))
    score = bp * math.exp(sum(log_precisions) / max_n) * 100.0
    return round(score, 2)

def compute_chrf(hypotheses, references, n_char=6, beta=2.0):
    """
    Computes character n-gram F-score (chrF), a standard metric for low-resource
    and morphologically rich languages where exact word matches are sparse.
    """
    smooth = 1e-9
    f_scores = []

    for hyp, ref in zip(hypotheses, references):
        h_chars = list(hyp.replace(" ", ""))
        r_chars = list(ref.replace(" ", ""))

        if not h_chars or not r_chars:
            continue

        n_f = []
        for n in range(1, n_char + 1):
            h_ngrams = Counter([tuple(h_chars[i:i+n]) for i in range(len(h_chars) - n + 1)])
            r_ngrams = Counter([tuple(r_chars[i:i+n]) for i in range(len(r_chars) - n + 1)])

            matches = sum(min(cnt, r_ngrams.get(ng, 0)) for ng, cnt in h_ngrams.items())
            prec = (matches + smooth) / (max(len(h_chars) - n + 1, 0) + smooth)
            rec = (matches + smooth) / (max(len(r_chars) - n + 1, 0) + smooth)

            f = (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec + smooth)
            n_f.append(f)

        if n_f:
            f_scores.append(sum(n_f) / len(n_f))

    avg_chrf = (sum(f_scores) / max(len(f_scores), 1)) * 100.0
    return round(avg_chrf, 2)

def compute_ter(hypotheses, references):
    """
    Translation Edit Rate (TER) approximation based on Levenshtein token edit distance.
    Lower is better.
    """
    def edit_distance(s1, s2):
        dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        for i in range(len(s1) + 1):
            dp[i][0] = i
        for j in range(len(s2) + 1):
            dp[0][j] = j
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[len(s1)][len(s2)]

    total_edits = 0
    total_ref_words = 0
    for hyp, ref in zip(hypotheses, references):
        h_words = hyp.strip().split()
        r_words = ref.strip().split()
        edits = edit_distance(h_words, r_words)
        total_edits += edits
        total_ref_words += max(len(r_words), 1)

    ter = (total_edits / max(total_ref_words, 1)) * 100.0
    return round(min(ter, 100.0), 2)

def evaluate_all_metrics(hypotheses, references):
    return {
        "bleu": compute_bleu(hypotheses, references),
        "chrf": compute_chrf(hypotheses, references),
        "ter": compute_ter(hypotheses, references),
        "total_test_samples": len(hypotheses)
    }
