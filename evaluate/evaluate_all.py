import sys
from evaluate_present import get_list
from collections import Counter
from nltk.stem.porter import PorterStemmer
import pickle

stemmer = PorterStemmer()            

def norm(container):
    '''消除每一个元素的空格，方便比较'''
    try:
        remove_space_set = set(["".join(w.split(" ")) for w in container])
        return remove_space_set
    except AttributeError:
        print(container)
        print("AttributeError")


def stem_norm(container):
    '''消除每一个元素的空格，方便比较，同时做stem处理'''
    # 去除none
    container = set(container) - set(['none'])
    result_set = set()
    remove_space_set = set([w.strip() for w in container])
    for keyphrase in remove_space_set:
        stem_keyphrase_list = []
        for word in keyphrase.split(" "):
            stem_keyphrase_list.append(stemmer.stem(word))
        result_set.add(" ".join(stem_keyphrase_list))
    return result_set

def restore_article(input_words):
    article = []
    cur_word = ""
    for w in input_words:
        if w[:2] == '##':
            cur_word += w[2:]
        elif len(cur_word) > 0:
            article.append(cur_word)
            cur_word = w
        else:
            cur_word = w
    if len(cur_word) > 0:
        article.append(cur_word)
    return " " + " ".join(article) + " "

pred_absent = sys.argv[1]
gold_absent = sys.argv[2]
pred_present = sys.argv[3]
input_words = sys.argv[4]
gold_present = sys.argv[5]
path_prob = sys.argv[6]

prob_list = pickle.load(open(path_prob, "rb"))

# 统计absent部分
gold_absent_list = []
pred_absent_list = []
gold_present_list = []

with open(pred_absent, "r") as fp:
    for line in fp:
        parts = line.strip().split(' ; ')
        k_set =set()
        for i in parts:
            k_set.add(i)
        pred_absent_list.append(stem_norm(k_set))

with open(gold_absent, "r") as fg:
    for line in fg:
        parts = line.strip().split(' ; ')
        k_abs_set = set()
        for i in parts:
            k_abs_set.add(i.replace(' ##', ''))
        gold_absent_list.append(stem_norm(k_abs_set))

with open(gold_present, "r") as fg:
    for line in fg:
        k_pre_set = stem_norm(line.strip().split('@'))
        gold_present_list.append(k_pre_set)

# 统计present部分
pred_present_list = []

for input_words, gold_full_list, predict_label, all_label_phrase, predict_prob, topk_phrase in get_list(pred_present, input_words, gold_present, prob_list):
     pred_present_list.append(stem_norm(all_label_phrase))

predict_count = 0
gold_count = 0
hit_count = 0
# model_name = path_prob.split(".")[-2]
# print(model_name)
# s = input()
for idx, (ga, pa, gp, pp) in enumerate(zip(gold_absent_list, pred_absent_list, gold_present_list, pred_present_list)):
    gold = ga | gp
    predict = pa | pp
    predict_count += len(predict)
    gold_count += len(gold)
    hit_count += len(predict & gold)
p = hit_count / predict_count
r = hit_count / gold_count
f = 2 * p * r / (p + r)
print("overall p@m r@m f@m")
print(round(p, 5), round(r, 5) ,round(f,5))
