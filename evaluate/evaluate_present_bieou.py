# coding:utf-8
import sys
from nltk.stem.porter import PorterStemmer
import argparse
import pickle

stemmer = PorterStemmer()            

def get_list(path_p, path_words, path_gold, prob_list):
    """根据标注结果提取出短语

    Args:
        path_p (str): test_results.txt路径， 模型预测的每个label
        path_words (str): seq.in路径，输入文字
        path_gold (str): .keyphrase或者.present.test路径，ground truth present keyphrase
        prob_list (list): .label.prob加载出来的标注概率list

    Returns:
        yield 输入文章、ground truth label、预测标签、模型预测结果(f@5)、模型预测结果(f@m)、每个token预测标签的对应概率列表

    """
    predict_all = []
    gold_all = []
    input_all = []
    gold_tag_all = []
    
    file_words = open(path_words, "r", encoding="utf8")
    file_full_gold = open(path_gold, "r", encoding="utf8")
    file_predict = open(path_p, "r", encoding="utf8")

    # label 2 id
    label_list = ['O', 'B', 'I', 'X', 'E', 'O', 'U', '[S2S_CLS]', '[S2S_SEP]', '[SEP]', '[CLS]', '[S2S_SOS]']
    label2id = {label:idx for idx, label in enumerate(label_list)}

    for idx, (line_words, line_full_gold, line_p) in enumerate(zip(file_words, file_full_gold, file_predict)):
        line_words = line_words.strip().split(" ")
        # 删除第一个BERT添加的CLS
        line_p = line_p.strip().split(" ")[1:]
        # line_p = line_p.strip().split(" ")
       
        # 得到预测标签的概率, word_no + 1 除开开头的CLS
        predict_prob = [prob_list[idx][word_no + 1][label2id[label]] for word_no, label in enumerate(line_p)]
        # predict_prob = [prob_list[idx][word_no + 1][min(8, label2id[label])] for word_no, label in enumerate(line_p)]
        # predict_prob = []
        # for word_no, label in enumerate(line_p):
        #     try:
        #         tmp = prob_list[idx][word_no+1][label2id[label]] 
        #     except IndexError:
        #         print(idx, word_no+1, label2id[label])
        #         print(len(prob_list))
        #         print(len(prob_list[idx]))
        #         print(len(prob_list[idx][word_no+1]))
        #         s = input()
        #     predict_prob.append(tmp)
        

        # 处理文中标注部分
        cur = 0
        all_label_phrase_score = []
        cur_phrase_score = []
        all_label_phrase = []
        cur_phrase = []

        # line_words即输入文件，是还没有经过padding或者截断的，而line_p是预测标签，是统一长度的
        # 因此我们按较小者长度遍历
        min_l = min(len(line_p), len(line_words))
        while cur < min_l:
            if len(cur_phrase) > 0:
                all_label_phrase.append(" ".join(cur_phrase))
                all_label_phrase_score.append(sum(cur_phrase_score) / len(cur_phrase_score))
                cur_phrase_score = []
                cur_phrase = []
            if line_p[cur] == "B":
                cur_word = line_words[cur]
                cur_phrase = []
                cur_phrase_score = [predict_prob[cur]]
                cur += 1
                while cur < min_l and (line_p[cur] == "I" or line_p[cur] == "E"):
                    if line_words[cur][:2] != "##":
                        cur_phrase.append(cur_word)
                        cur_word = line_words[cur]
                    elif line_p[cur] == "E":
                        cur_word += line_words[cur].lstrip('##')
                        cur_phrase.append(cur_word)
                        cur_word = ""
                    else:
                        cur_word += line_words[cur].lstrip('##')
                    cur_phrase_score.append(predict_prob[cur])
                    cur += 1
                cur_phrase.append(cur_word)
            elif line_p[cur] == "U":
                cur_phrase.append(line_words[cur])
                cur_phrase_score.append(predict_prob[cur])
                cur += 1
            else:
                cur += 1
            
        if len(cur_phrase) > 0:
            all_label_phrase.append(" ".join(cur_phrase))
            all_label_phrase_score.append(sum(cur_phrase_score) / len(cur_phrase_score))
            cur_phrase_score = []
            cur_phrase = []

        line_full_gold = line_full_gold.strip().split('@')
        
        # 根据phrase标注概率计算得分，多处标注取最大的得分
        phrase_score_dict = dict()
        for phrase, score in zip(all_label_phrase, all_label_phrase_score):
            if phrase in phrase_score_dict:
                phrase_score_dict[phrase] = max(phrase_score_dict[phrase], score)
            else:
                phrase_score_dict[phrase] = score
        # 根据得分取top5计算f1@5
        phrase_score_sorted = sorted(phrase_score_dict.items(), key = lambda kv:-kv[1])
        topk_phrase = [item[0] for item in phrase_score_sorted[:5]]

        # 生成器模式
        yield line_words, line_full_gold, line_p, all_label_phrase, predict_prob, topk_phrase

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

if __name__ == "__main__":

    # 预测标签路径、输入序列路径、完整gold keyphrase路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--stem_evaluate', action='store_true', default=False, help="whether use stem before evaluation, default: False")
    parser.add_argument('--stem_divide', action='store_true', default=False, help="whether use stem before dividing keyphrases into present and absent parts, default: False")
    parser.add_argument('--label_path', help='the path to the predict .label file')
    parser.add_argument('--input_path', help='the path to the input .seq.in file')
    parser.add_argument('--gold_path', help='the path to the ground truth file, if stem_divide, it is .present.test file; else, it is .keyphrase file')
    parser.add_argument('--prob_path', help='the path to the .prob file which records the predicted label probs for calculating present f1@5')
    args = parser.parse_args()
    print(args)
    path_p = args.label_path
    path_words = args.input_path
    path_gold = args.gold_path
    path_prob = args.prob_path
    stem_evaluate = args.stem_evaluate
    stem_divide = args.stem_divide

    if (".present.test" in path_gold and stem_divide) or (".keyphrase" in path_gold and not stem_divide):
        pass
    else:
        sys.exit("Error, when stem_divide we use .present.test(which is provided by meng) as ground truth; when not stem_divide we use .keyphrase(contain all keyphrases and we divide by ourselves) as ground truth. please check your settings")
         

    # 读取输出概率
    prob_list = pickle.load(open(path_prob, "rb"))
    # 连续出现gold keyphrase
    continuous_gold_count = 0
    # 文中连续标注出的数量
    article_label_count = 0
    # 连续出现命中量
    continuous_hit = 0
    # 文中连续标注出的数量(top5)
    article_label_count_topk = 0
    # 连续出现命中量(top5)
    continuous_hit_topk = 0
    # 没有预测结果的样本数
    empty_count = 0
    count = 0
 
    for idx, (input_words, gold_full_list, predict_label, all_label_phrase, predict_prob, topk_phrase) in enumerate(get_list(path_p, path_words, path_gold, prob_list)):
        # 从完整gold keyphrase集合中筛选连续出现短语gold keyphrase集合
        article = restore_article(input_words)
        article_vocab = set(article.split(" "))
        continuous_set = set()
        for gold_keyphrase in gold_full_list:
            # 对替换数字做处理 <digit> --> < digit >
            if "<digit>" in gold_keyphrase:
                gold_keyphrase = gold_keyphrase.replace("<digit>", "< digit >")
            
            if stem_divide:
                # 直接用.present结果，即meng区分的present
                continuous_set.add(gold_keyphrase)
            else:
                # 判断keyphrase是否在原文中出现，首尾加空格区分边界
                if " " + gold_keyphrase + " " in article:
                    continuous_set.add(gold_keyphrase)
        
        # 更新统计量
        continuous_gold_count += len(continuous_set)
        article_label_count += len(set(all_label_phrase))
        article_label_count_topk += len(set(topk_phrase))

        # 计算命中之前是否stem
        if stem_evaluate:
            continuous_hit += len(stem_norm(continuous_set) & stem_norm(all_label_phrase))
            continuous_hit_topk += len(stem_norm(continuous_set) & stem_norm(topk_phrase))
        else:
            continuous_hit_topk += len(norm(continuous_set) & norm(topk_phrase))
            continuous_hit += len(norm(continuous_set) & norm(all_label_phrase))
    
        # # 检查
        # print(idx)
        # print("\n所有gold keyphrase")
        # print(gold_full_list)
        # print("\n文中连续出现的gold keyphrase")
        # print(continuous_set)
        # print("\n模型预测结果（文中标注连续出现）")
        # print(set(all_label_phrase))
        # print("\n差别")
        # print(norm(continuous_set) - norm(all_label_phrase))
        # print("\n文章")
        # print(" ".join(input_words))
        # print("\n还原文章")
        # print(article)
        # print("\n文章及其标注结果")
        # print([[word,label] for word, label in zip(input_words, predict_label)])
        # print("\n"*2)
        # s = input()

    # 连续出现prf值
    p = continuous_hit / article_label_count
    r = continuous_hit / continuous_gold_count
    f = 2 * p * r / (p + r)
    print("present p,r,f@m")
    print(round(p, 4), round(r, 4), round(f, 4))

    p = continuous_hit_topk / article_label_count_topk
    r = continuous_hit_topk / continuous_gold_count
    f = 2 * p * r / (p + r)
    print("present p,r,f@5")
    print(round(p, 4), round(r, 4), round(f, 4)) 
