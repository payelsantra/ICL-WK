import pickle
import pandas as pd
import re
import random
from collections import OrderedDict
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict

def define_k_shot(X_test_ID,all_dict_id,num_few):
    superstring=OrderedDict()
    string1="Given that"
    for i in X_test_ID:
        string2=" "
        for k,l in enumerate(all_dict_id[i]["evidence"][:num_few]):
            neigh_claim=l
            neigh_claim_split=neigh_claim.split(".")
            cl_final=neigh_claim_split[0]
            if k!=num_few-1:
                string2 += f'{cl_final}' + ", "
            else:
                string2 += " and "+ f'{cl_final}' + ", "
            string3=string1+string2
        superstring[i]=string3
    return superstring

training_data = pd.read_csv("unique_claim_train_set.csv")
test_data = pd.read_csv("unique_claim_test_set.csv")
X_test=test_data["CLAIM"]
y_test=test_data["LABEL"]
X_test_ID=test_data["ID"]

file1 = open("fever.tsv")
file1_ls=file1.readlines()

all_dict_id={}
for lines in file1_ls:
    line_id_=lines.strip("\n").split("\t")[0]
    line_claim_=lines.strip("\n").split("\t")[1]
    line_string=lines.strip("\n").split("\t")[2:]
    if len(line_string)==0:
        #print(line_id_)
        random_line_string=="null"
    else:
        random_line_string=line_string
    all_dict_id[int(line_id_)]={"claim":line_claim_}
    all_dict_id[int(line_id_)].update({"evidence":random_line_string})

#data for 4-shot
superstring=define_k_shot(X_test_ID,all_dict_id,4)
zip_df_train = pd.DataFrame(zip(list(superstring.keys()),X_test,list(superstring.values()),y_test),columns=["id","cllaim","string",'label'])
zip_df_train.to_csv('data_top4.csv')
