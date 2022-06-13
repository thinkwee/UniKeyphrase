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
        for i in parts[:5]:
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
     # 从完整gold keyphrase集合中筛选连续出现短语gold keyphrase集合
     pred_present_list.append(stem_norm(all_label_phrase))

def keyphrase_to_bw(keyphrase):
    # set to list
    bw_list = [item for item in keyphrase if len(item)>0]
    return Counter(bw_list)

error_bag_of_words = 0
count = 0
for idx, (ga, pa, gp, pp) in enumerate(zip(gold_absent_list, pred_absent_list, gold_present_list, pred_present_list)):
    # 统计pred和gold的 present-absent的词袋
    # 再统计这两个词袋之差
    diff_gold = keyphrase_to_bw(gp)
    diff_gold.subtract(keyphrase_to_bw(ga))
    
    diff_pred = keyphrase_to_bw(pp)
    diff_pred.subtract(keyphrase_to_bw(pa))
    diff_pred.subtract(diff_gold)
    error_bag_of_words += sum([abs(diff_pred[key]) for key in diff_pred])

    # # 统计pred和gold的 present + absent的词袋
    # # 再统计这两个词袋之差
    # total_gold = keyphrase_to_bw(gp) + keyphrase_to_bw(ga)
    # total_pred = keyphrase_to_bw(pp) + keyphrase_to_bw(pa)
    # total_pred.subtract(total_gold)
    # error_bag_of_words += sum([abs(total_pred[key]) for key in total_pred])
    count += 1

print("diff error %f" % (error_bag_of_words / count))
