import pickle
import sys

name_prefix = sys.argv[1]
name_suffix = sys.argv[2]
name_output = sys.argv[3]

prob_list = []
for i in range(10):
    tmp_list = pickle.load(open(name_prefix + str(i) + name_suffix + ".prob", "rb"))
    prob_list += tmp_list
pickle.dump(prob_list, open(name_output, "wb"))
    
