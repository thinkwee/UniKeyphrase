import os
from bert import tokenization
from tqdm import tqdm
import argparse

def process(json_file_path, output_path, divide_stem):
    print(json_file)
    total_count = 0
    include_count = 0
    noise_count = 0
    data_type = json_file.split("_")[1]
    assert data_type in {"train", "test", "valid"}

    # 写文件，分别是输入token序列、输出标签序列、完整kephrase序列、absent keyphrase序列、present keyphrase序列、用于测试的present keyphrase序列
    # 测试的present keyphrase序列是指还没有去除嵌套keyphrase
    fw_in = open(output_path + "/" + dataset + "." + data_type + ".seq.in", "w")
    fw_out = open(output_path + "/" + dataset + "." + data_type + ".seq.out", "w")
    fw_gold_keyphrase = open(output_path + "/" + dataset + "." + data_type + ".keyphrase", "w")
    fw_absent = open(output_path + "/" + dataset + "." + data_type + ".absent", "w")
    fw_present = open(output_path + "/" + dataset + "." + data_type + ".present", "w")
    fw_present_test = open(output_path + "/" + dataset + "." + data_type + ".present.test", "w")

    with open("./" + dataset + "/" + json_file,"r") as f:
        for idx, line in tqdm(enumerate(f)):
            json_item = eval(line)
            # 输入token
            content = ' '.join(json_item['meng17_tokenized']['src']).lower()
            # prsent keyphrase
            present = json_item['meng17_tokenized']['present_tgt']
            present_set = set([" ".join(l) for l in present]) - set([''])
            # absent keyphrase
            absent = json_item['meng17_tokenized']['absent_tgt']
            absent_set = set([" ".join(l) for l in absent]) - set([''])
            # 存一份最原始的所有keyphrase
            all_set = present_set | absent_set
            # 存一份没有过滤嵌套的present用于测试
            fw_present_test.write("@".join(present_set) + "\n")

            if not divide_stem:
                # 默认的是meng给出的present absent区分，即以keyphrase stem之后是否在stem后的文章中区分
                # 假如不做stem，即直接匹配，连续出现在文中的为present keyphrase
                present_set = set()
                absent_set = set()
                for keyphrase in all_set:
                    keyphrase = keyphrase.lower()
                    if keyphrase in content:
                        present_set.add(keyphrase)
                    else:
                        absent_set.add(keyphrase)

            # 找出最长标签，防止重复标注
            # TODO: 目前直接用子串判断是否嵌套，但是这样会多判断一些嵌套
            # 由于subword的影响，还没有完美的判断嵌套方案
            longest_keyphrase = []
            for keyphrase in present_set:
                left_set = present_set - set([keyphrase])
                include_flag = False
                for left_keyphrase in left_set:
                    if keyphrase in left_keyphrase:
                        include_flag = True
                        break
                if not include_flag:
                    longest_keyphrase.append(keyphrase)
                else:
                    include_count += 1
                total_count += 1
            present_set = set([kp.strip() for kp in longest_keyphrase])

            # 在正文中标注出gold keyphrase
            # 具体是用特殊边界@#^来围住标注区间
            for continuous_keyphrase in present_set:
                content = content.replace(continuous_keyphrase, "@#^#tag_is_here#" + continuous_keyphrase + "@#^")

            # 记录tokenize之后的wordpiece及其label
            tokens = []
            label_list = []

            # 之后根据特殊边界把原始文章分成一些span
            for span in content.split("@#^"):
                # 假如是keyphrase的span
                if "#tag_is_here#" in span:
                    first = True
                    # 对keyphrase里每个词，切分为subword，假如是keyphrase第一个词，则标为BX... ，否则标为IX....
                    for word in span[13:].split(" "):
                        tokens_word = tokenizer.tokenize(word)
                        tokens += tokens_word
                        if first:
                            mark_label = 'B'
                            first = False
                        else:
                            mark_label = 'I'
                        label_list += [mark_label] + ['X'] * (len(tokens_word) - 1)
                # 假如是普通的文本span，直接分词，并且全标O
                else:
                    tokens_span = tokenizer.tokenize(span)
                    tokens += tokens_span
                    label_list += ['O'] * len(tokens_span)

            # absent短语为空的话加入none
            if len(absent_set) == 0:
                absent_set.add("none")
            # 得到模型要生成的absent拼接文本，用分号分隔
            absent_str = " ; ".join([" ".join(tokenizer.tokenize(word)) for word in absent_set])
            
            # present和完整keyphrase用@分割，只是为了后续评价脚本用
            all_str = "@".join(all_set)
            if len(present_set) == 0:
                present_set.add("none")
            present_str = "@".join(present_set)

            # 假如文本存在噪声（乱码），可能切分后标签和token数量对不上，打印错误并计数
            if len(tokens) != len(label_list):
                print("!!  len(tokens) != len(label_list)  !!")
                print(len(tokens), len(label_list))
                print(idx)
                print(content)
                print(present_set)
                print(absent_set)
                for i in range(max(len(tokens), len(label_list))):
                    print([tokens[i], label_list[i]], end = " ")
                noise_count += 1
                continue

            # 写入文件
            fw_in.write(" ".join(tokens) + "\n")
            fw_out.write(" ".join(label_list) + "\n")
            fw_gold_keyphrase.write(all_str + "\n")
            fw_absent.write(absent_str + "\n")
            fw_present.write(present_str + "\n")

            # # 检查处理结果
            # print("\ninput tokens")
            # print(tokens)
            # print("\noutput labels")
            # print(label_list)
            # print("\nall keyphrase")
            # print(all_set)
            # print("\npresent keyphrase")
            # print(present_set)
            # print("\nabsent keyphrase")
            # print(absent_set)
            # print("\nabsent_str")
            # print(absent_str)
            # s = input()

    print("%d noise samples ignored" % noise_count)
    print("%f keyphrase is ignored because they are included other keyphrases" % (include_count / total_count)) 
    fw_in.close()
    fw_out.close()
    fw_gold_keyphrase.close()
    fw_absent.close()
    fw_present.close()
    fw_present_test.close()

parser = argparse.ArgumentParser()
parser.add_argument('--divide_stem', action='store_true', default=False, help="whether use stem before divide keyphrases into present and absent parts, default False")
parser.add_argument('--data_path', help='the path to the *_meng17token.json file, type "all" to process all datafiles')
parser.add_argument('--output_path', help='the output path to save all the files(seq.in seq.out .present .absent .keyphrase .present.test')

args = parser.parse_args()
print(args)

# 初始化tokenizer，因为要对齐token和label
# 因此在输入BERT之前就需要tokenize
tokenizer = tokenization.FullTokenizer(
    vocab_file="/apdcephfs/share_774517/data/thinkweeliu/unilm_data/bert_config/bert-base-cased/vocab.txt",   
    do_lower_case=True)

if args.data_path == "all":
    dataset_names = ['inspec', 'nus', 'semeval', 'kp20k']

    print("process all file")
    for dataset in dataset_names:
        print(dataset)
        for json_file in os.listdir("./" + dataset):
            if "meng17token" in json_file:
                process(json_file, args.output_path, args.divide_stem)
else:
    process(args.data_path, args.output_path, args.divide_stem)
