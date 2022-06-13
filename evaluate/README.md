# 计算指标
- evaluate_present.py和evaluate_absent.py分别计算序列标注present keyphrase和文本生成absent keyphrase的结果
- 提供了merge_prob.py文件，用于测试kp20k时合并.prob

# 参数
- present评价代码添加了stem开关和读取概率文件计算f1@5(需要在运行decode_seq2seq.py推断时打开output_prob开关）,参数说明如下：
  ```
  usage: evaluate_present.py [-h] [--stem_evaluate] [--stem_divide]
                             [--label_path LABEL_PATH] [--input_path INPUT_PATH]
                             [--gold_path GOLD_PATH] [--prob_path PROB_PATH]
  
  optional arguments:
    -h, --help            show this help message and exit
    --stem_evaluate       whether use stem before evaluation, default: False
    --stem_divide         whether use stem before dividing keyphrases into
                          present and absent parts, default: False
    --label_path LABEL_PATH
                          the path to the predict .label file
    --input_path INPUT_PATH
                          the path to the input .seq.in file
    --gold_path GOLD_PATH
                          the path to the ground truth file, if stem_divide, it
                          is .present.test file; else, it is .keyphrase file
    --prob_path PROB_PATH
                          the path to the .prob file which records the predicted
                          label probs for calculating present f1@5
  ```
- absent评价代码添加了stem开关，参数说明如下：
  ```
  usage: evaluate_absent.py [-h] [--stem_evaluate] [--stem_divide]
                           [--input_path INPUT_PATH] [--pred_path PRED_PATH]
                           [--gold_path GOLD_PATH]

  optional arguments:
    -h, --help            show this help message and exit
    --stem_evaluate       whether use stem before evaluation, default: False
    --stem_divide         whether use stem before dividing keyphrases into
                          present and absent parts, default: False
    --input_path INPUT_PATH
                          the path to the input .seq.in file
    --pred_path PRED_PATH
                          the path to the predict model.absent file
    --gold_path GOLD_PATH
                          the path to the ground truth file, if stem_divide, it
                          is .absent file; else, it is .keyphrase file
  ```

# 其他脚本
- evaluate_diff.py：测量词袋误差
- evaluate_present_bieou：present部分采用BIEOU标注方案的评测脚本
- evaluate_all.py：不区分present/absent的评测及其启动脚本
