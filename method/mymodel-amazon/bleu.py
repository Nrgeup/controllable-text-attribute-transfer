#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import copy

def bleu_count(hypothesis, references, max_n=4):
    ret_len_hyp = 0
    ret_len_ref = 0
    ret_clip_count = [0]*max_n
    ret_count = [0]*max_n
    for m in range(len(hypothesis)):
        hyp, ref = hypothesis[m], references[m]
        x = hyp.split()
        y = [r.split() for r in ref]
        x_len = len(x)
        y_len = [len(s) for s in y]
        n_ref = len(ref)

        closest_diff = 9999
        closest_length = 9999
        ref_ngram = dict()

        for i in range(n_ref):
            diff = abs(y_len[i]-x_len)
            if diff < closest_diff:
                closest_diff = diff
                closest_length = y_len[i]
            elif diff==closest_diff and y_len[i] < closest_length:
                closest_length = y_len[i]

            for n in range(max_n):
                sent_ngram = dict()
                for st in range(0, y_len[i]-n):
                    ngram = "%d"%(n+1)
                    for k in range(n+1):
                        j = st+k
                        ngram += " %s"%(y[i][j])
                    if ngram not in sent_ngram:
                        sent_ngram[ngram]=0
                    sent_ngram[ngram]+=1
                for ngram in sent_ngram.keys():
                    if ngram not in ref_ngram or ref_ngram[ngram]<sent_ngram[ngram]:
                        ref_ngram[ngram] = sent_ngram[ngram]

        ret_len_hyp += x_len
        ret_len_ref += closest_length

        for n in range(max_n):
            hyp_ngram = dict()
            for st in range(0, x_len-n):
                ngram = "%d"%(n+1)
                for k in range(n+1):
                    j = st+k
                    ngram += " %s"%(x[j])
                if ngram not in hyp_ngram:
                    hyp_ngram[ngram]=0
                hyp_ngram[ngram]+=1
            for ngram in hyp_ngram.keys():
                if ngram in ref_ngram:
                    ret_clip_count[n] += min(ref_ngram[ngram], hyp_ngram[ngram])
                ret_count[n] += hyp_ngram[ngram]

    return ret_clip_count, ret_count, ret_len_hyp, ret_len_ref


def corpus_bleu(hypothesis, references, max_n=4):
    assert(len(hypothesis) == len(references))
    clip_count, count, total_len_hyp, total_len_ref = bleu_count(hypothesis, references, max_n=max_n)
    brevity_penalty = 1.0
    bleu_scores = []
    bleu = 0
    for n in range(max_n):
        if count[n]>0:
            bleu_scores.append(clip_count[n]/count[n])
        else:
            bleu_scores.append(0)
    if total_len_hyp < total_len_ref:
        if total_len_hyp==0:
            brevity_penalty = 0.0
        else:
            brevity_penalty = math.exp(1 - total_len_ref/total_len_hyp)
    def my_log(x):
        if x == 0:
            return -9999999999.0
        elif x < 0:
            raise Exception("Value Error")
        return math.log(x)
    log_bleu = 0.0
    for n in range(max_n):
        log_bleu += my_log(bleu_scores[n])
    bleu = brevity_penalty*math.exp(log_bleu / float(max_n))
    return [bleu]+bleu_scores, [brevity_penalty, total_len_hyp/total_len_ref, total_len_hyp, total_len_ref]


def incremental_bleu_count(hypothesis, references, max_n=4):
    ret_len_hyp = []
    ret_len_ref = []
    ret_clip_count = []
    ret_count = []
    for m in range(len(hypothesis)):
        hyp, ref = hypothesis[m], references[m]
        x = hyp.split()
        y = [r.split() for r in ref]
        x_len = len(x)
        y_len = [len(s) for s in y]
        n_ref = len(ref)

        ref_ngram = dict()

        for i in range(n_ref):
            for n in range(max_n):
                sent_ngram = dict()
                for st in range(0, y_len[i]-n):
                    ngram = "%d"%(n+1)
                    for k in range(n+1):
                        j = st+k
                        ngram += " %s"%(y[i][j])
                    if ngram not in sent_ngram:
                        sent_ngram[ngram]=0
                    sent_ngram[ngram]+=1
                for ngram in sent_ngram.keys():
                    if ngram not in ref_ngram or ref_ngram[ngram]<sent_ngram[ngram]:
                        ref_ngram[ngram] = sent_ngram[ngram]
        y_len = sorted(y_len)
        ret_len_hyp.append([])
        ret_len_ref.append([])
        ret_clip_count.append([])
        ret_count.append([])

        hyp_ngram = dict()
        p_closest = 0
        for i in range(x_len):
            if i == 0:
                ret_clip_count[-1].append([0]*max_n)
                ret_count[-1].append([0]*max_n)
            else:
                ret_clip_count[-1].append(copy.deepcopy(ret_clip_count[-1][-1]))
                ret_count[-1].append(copy.deepcopy(ret_count[-1][-1]))

            j = i+1
            ret_len_hyp[-1].append(i+1)
            if j>y_len[p_closest]:
                while j>y_len[p_closest] and p_closest<n_ref-1:
                    p_closest+=1
            tmp_closest_diff = 9999
            tmp_closest_len = 9999
            if p_closest>0 and (j-y_len[p_closest-1])<tmp_closest_diff:
                tmp_closest_diff=j-y_len[p_closest-1]
                tmp_closest_len = y_len[p_closest-1]
            if p_closest<n_ref and (y_len[p_closest]-j)<tmp_closest_diff:
                tmp_closest_diff=y_len[p_closest]-j
                tmp_closest_len = y_len[p_closest]

            ret_len_ref[-1].append(tmp_closest_len)
            for n in range(max_n):
                st = i-n
                if st>=0:
                    ngram = "%d"%(n+1)
                    for k in range(n+1):
                        j = st+k
                        ngram += " %s"%(x[j])
                    if ngram not in hyp_ngram:
                        hyp_ngram[ngram]=0
                    hyp_ngram[ngram]+=1
                    ret_count[-1][-1][n] += 1
                    if ngram in ref_ngram  and hyp_ngram[ngram]<=ref_ngram[ngram]:
                        ret_clip_count[-1][-1][n] += 1

    return ret_clip_count, ret_count, ret_len_hyp, ret_len_ref

def incremental_sent_bleu(hypothesis, references, max_n=4):
    clip_count, count, total_len_hyp, total_len_ref = incremental_bleu_count([hypothesis], [references], max_n=max_n)
    clip_count = clip_count[0]
    count = count[0]
    total_len_hyp = total_len_hyp[0]
    total_len_ref = total_len_ref[0]
    n_len = len(clip_count)
    ret = []
    for i in range(n_len):
        brevity_penalty = 1.0
        bleu_scores = []
        bleu = 0
        for n in range(max_n):
            if count[i][n]>0:
                bleu_scores.append(clip_count[i][n]/count[i][n])
            else:
                bleu_scores.append(0)
        if total_len_hyp[i] < total_len_ref[i]:
            if total_len_hyp[i]==0:
                brevity_penalty = 0.0
            else:
                brevity_penalty = math.exp(1 - total_len_ref[i]/total_len_hyp[i])
        def my_log(x):
            if x == 0:
                return -9999999999.0
            elif x < 0:
                raise Exception("Value Error")
            return math.log(x)
        log_bleu = 0.0
        for n in range(max_n):
            log_bleu += my_log(bleu_scores[n])
        bleu = brevity_penalty*math.exp(log_bleu / float(max_n))
        ret.append(bleu)
    return ret

def incremental_test_corpus_bleu(hypothesis, references, max_n=4):
    assert(len(hypothesis) == len(references))
    tmp_clip_count, tmp_count, tmp_total_len_hyp, tmp_total_len_ref = incremental_bleu_count(hypothesis, references, max_n=max_n)
    clip_count = [0]*4
    count = [0]*4
    total_len_hyp = 0
    total_len_ref = 0
    for i in range(len(hypothesis)):
        for n in range(4):
            clip_count[n]+=tmp_clip_count[i][-1][n]
            count[n] += tmp_count[i][-1][n]
        total_len_hyp += tmp_total_len_hyp[i][-1]
        total_len_ref += tmp_total_len_ref[i][-1]
    brevity_penalty = 1.0
    bleu_scores = []
    bleu = 0
    for n in range(max_n):
        if count[n]>0:
            bleu_scores.append(clip_count[n]/count[n])
        else:
            bleu_scores.append(0)
    if total_len_hyp < total_len_ref:
        if total_len_hyp==0:
            brevity_penalty = 0.0
        else:
            brevity_penalty = math.exp(1 - total_len_ref/total_len_hyp)
    def my_log(x):
        if x == 0:
            return -9999999999.0
        elif x < 0:
            raise Exception("Value Error")
        return math.log(x)
    log_bleu = 0.0
    for n in range(max_n):
        log_bleu += my_log(bleu_scores[n])
    bleu = brevity_penalty*math.exp(log_bleu / float(max_n))
    return [bleu]+bleu_scores, [brevity_penalty, total_len_hyp/total_len_ref, total_len_hyp, total_len_ref]

