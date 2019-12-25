# coding: utf-8
######### EVALUATE.PY ########
# Evaluation script for NLP project, 4 year.
# Input: gold standard files in `gold` folder
# Input: testing results in `scored` folder. Files should be named like [text id number].* (see code below)
# Input format (test): 4-column file, both ',' and '-'-separated token ids are accepted.
# Input format (GS): 5-column file.
# Scoring assumptions:
# 1. Each span intersecting with golden is counted as a match.
# 2. Recall is calculated as (num matches) / (golden size)
# 3. Precision is calculated as (num matches) / (test size)

import os
import sys
from collections import defaultdict


class SentimentSpan:
    is_gold = False
    agr = False
    
    def __init__(self, line):
        line = line.split('\t')
        if len(line) == 5:
            self.is_gold = True
            if ',' in line[0]:
                self.agr = True
            self.sent_id = int(line[1])
            self.token_id = list(map(int, line[2].split(',')))
            self.aspect = line[3].lower()
            self.sentiment = line[4]
        else:
            line = '\t'.join(line)
            line = line.replace(', ', ',').replace(' ', '\t').replace('\t\t', '\t')
            line = line.split('\t')
            if len(line) != 4:
                print(line, file=sys.stderr)
                self.sent_id = None
                return
            self.sent_id = int(line[0])
            self.token_id = []
            if '-' in line[1]:
                start, *_, end = line[1].split('-')
                if start > end:
                    tmp = start
                    start = end
                    end = tmp
                self.token_id = list(range(int(start), int(end) + 1))
            else:
                self.token_id = list(map(int, line[1].split(',')))
            self.aspect = line[2].lower()
            self.sentiment = line[3]
    
    def intersects(self, other):
        if set(self.token_id) & set(other.token_id):
            return True
        return False
    
    def join(self, other):
        return list(set(self.token_id) | set(other.token_id))

    def is_similar(self, other):
        if self.sent_id is None:
            return False
        if other.sent_id is None:
            return False
        if not self.aspect or not other.aspect:
            return self.sentiment == other.sentiment
        return self.aspect == other.aspect and self.sentiment == other.sentiment
    
    def __repr__(self):
        return '\t'.join((
            str(self.sent_id),
            ','.join(map(str, sorted(self.token_id))),
            self.aspect,
            self.sentiment
        ))


if __name__ == '__main__':
    
    test_markups = {}
    for fn in os.listdir('scored'):
        id_ = fn.split('.')[0]
        print("PROCESSING:", id_, file=sys.stderr)
        test_markups[id_] = defaultdict(list)
        with open(os.path.join('scored', fn)) as f:
            for line in f:
                s = SentimentSpan(line.rstrip('\r\n'))
                test_markups[id_][s.sent_id].append(s)


    gold_markups = {}
    for fn in os.listdir('gold'):
        id_ = fn.split('-')[0]
        gold_markups[id_] = defaultdict(list)
        with open(os.path.join('gold', fn)) as f:
            for line in f:
                s = SentimentSpan(line.rstrip('\r\n'))
                gold_markups[id_][s.sent_id].append(s)

    # RECALL
    gold_ss = 0.
    test_ss = 0.
    added = []
    for i in test_markups:
        for gold_sent in gold_markups[i]:
            for s_gold in gold_markups[i][gold_sent]:
                if not s_gold.agr:
                    continue
                gold_ss += 1
                for s_test in test_markups[i][gold_sent]:
                    if s_gold.is_similar(s_test) and s_gold.intersects(s_test) and s_gold not in added:
                        added.append(s_gold)
                        test_ss += 1
    print("Found %d golden spans and %d similar test spans." % (gold_ss, test_ss))
    print("RECALL:", test_ss / gold_ss)

    # PRECISION
    gold_ss = 0.
    test_ss = 0.
    added = []
    for i in test_markups:
        for test_sent in test_markups[i]:
            for s_test in test_markups[i][test_sent]:
                test_ss += 1
                for s_gold in gold_markups[i][test_sent]:
                    if s_gold.is_similar(s_test) and s_gold.intersects(s_test) and s_test not in added:
                        added.append(s_test)
                        gold_ss += 1
    # Some debugging
    #             if s_test not in added:
    #                 print(i)
    #                 print(s_test)
    #                 print(gold_markups[i])
    print("Found %d test spans and %d similar golden spans." % (test_ss, gold_ss))
    print("PRECISION:", gold_ss / test_ss)
