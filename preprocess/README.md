# 数据预处理
- 运行start_make.sh，该样例展示了处理所有数据集
- 处理KP20k作者的预处理数据（meng17_tokenized）为输入序列(.seq.in)、输出标签(.seq.out)、absent keyphrase(.absent)、present keyphrase(.present)、未处理嵌套用于测试的present keyphrase(.present.test)共五个文件
- 需自行下载google bert代码到此目录下(./bert/)
- 增加了是否根据stem区分选项

# 参数
- 参数说明如下:
  ```
  usage: make_data_final.py [-h] [--divide_stem] [--data_path DATA_PATH]
                          [--output_path OUTPUT_PATH]

  optional arguments:
    -h, --help            show this help message and exit
    --divide_stem         whether use stem before divide keyphrases into present
                          and absent parts, default False
    --data_path DATA_PATH
                          the path to the *_meng17token.json file, type "all" to
                          process all datafiles
    --output_path OUTPUT_PATH
                          the output path to save all the files(seq.in seq.out
                          .present .absent .keyphrase .present.test
  ```
