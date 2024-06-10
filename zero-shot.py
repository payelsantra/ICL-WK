import argparse
from math import ceil
import os
import shutil
from typing import List
import inspect
from collections import namedtuple
from csv import DictReader
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
import numpy
import pickle as pkl

from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt.utils.reproduciblity import set_seed
use_cuda = True
set_seed(123)
#load_model
plm, tokenizer, model_config, WrapperClass = load_plm("flan-t5","google/flan-t5-large")


def load_dataset(input_datafile_name):
	with open(input_datafile_name, 'r') as f:
		dict_reader = DictReader(f)
		list_of_dict = list(dict_reader)
	dataset = {}
	dataset["test"] = []
	for data in list_of_dict:
	    input_example = InputExample(text_a = data['string'], text_b = data['cllaim'], label=int(data['label']), guid=data['id'])
	    dataset["test"].append(input_example)
	return dataset,list_of_dict

def evaluate(prompt_model, dataloader, desc,file1):
    prob_dict={}
    prompt_model.eval()
    allpreds = []
    alllabels = []
    all=[]
    pbar = tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        probability=numpy.exp(logits.cpu().numpy())
        for i in probability:
            if 'false' in prob_dict:
                prob_dict['false'].append(i[0])
            else:
                prob_dict['false']=[]
                prob_dict['false'].append(i[0])
            #print(i)
            if 'true' in prob_dict:
                prob_dict['true'].append(i[1])
            else:
                prob_dict['true']=[]
                prob_dict['true'].append(i[1])
                #print(f"|{i[0]:.2%}")
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    all.append(allpreds)
    all.append(alllabels)
    f1_sc=f1_score(alllabels, allpreds, average='macro')
    accur=accuracy_score(alllabels, allpreds)
    report =classification_report(alllabels, allpreds)
    tn, fp, fn, tp = confusion_matrix(alllabels, allpreds).ravel()
    file1.write(str(report)+ '\n')
    file1.write("\n")
    return f1_sc,accur,report,all,tn, fp, fn, tp,prob_dict

def find_accuracy(templeate,list_of_dict,dataset_to_pass,texts,classification_report_file_path,variable,out_path):
    file1 = open(classification_report_file_path,"w")
    accuracy_={}
    f1_score_={}
    truth={}
    for j,i in enumerate(templeate):
      print(i)
      classes = [ "False","True" ]
      promptVerbalizer = ManualVerbalizer(classes = classes,label_words = {"False": ['negative'],"True": ['positive']},tokenizer = tokenizer,)
      promptModel = PromptForClassification(template = i,plm = plm,  freeze_plm=True,verbalizer = promptVerbalizer,)
      test_dataloader = PromptDataLoader(dataset=dataset_to_pass, template=i, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,batch_size=16,shuffle=False, teacher_forcing=False, predict_eos_token=False, truncate_method="tail")
      promptModel = promptModel.cuda()
      fl_p=open(out_path,"wb")
      f1,ac,report_,all_,tn, fp, fn, tp,prob_dict_= evaluate(promptModel, test_dataloader, desc="Test",file1=file1)
      for data_test in list_of_dict:
        claim=data_test["cllaim"]
        id=data_test["id"]
        if 'claim' in prob_dict_:
            prob_dict_['claim'].append(claim)
        else:
            prob_dict_['claim']=[]
            prob_dict_['claim'].append(claim)
        if 'id' in prob_dict_:
            prob_dict_['id'].append(id)
        else:
            prob_dict_['id']=[]
            prob_dict_['id'].append(id)
      pkl.dump(all_,fl_p)
      print("f1",f1)
      print("ac",ac)
      accuracy_[texts[j]]=ac
      f1_score_[texts[j]]=f1
      truth[texts[j]]=(tn, fp, fn, tp)
    file1.close()
    return accuracy_,f1_score_,truth,prob_dict_

def make_csv(accuracy_,f1_score_,truth,prob_dict_,output_csv_result,output_csv_probability_score):
    df_result = pd.DataFrame(list(zip(accuracy_.keys(), accuracy_.values(),f1_score_.values(),truth.values())),columns =['temp', 'accuracy','f1_score','truth_level'])
    df_prob = pd.DataFrame(list(zip(prob_dict_['id'], prob_dict_['claim'],prob_dict_['true'],prob_dict_['false'])),columns =['id', 'claim','true','false'])
    df_result.to_csv(output_csv_result)
    df_prob.to_csv(output_csv_probability_score)
    return df_result,df_prob

def make_template(texts):
    my_template=[]
    #print(texts)
    for i in texts:
        temp=ManualTemplate(tokenizer=tokenizer, text=i)
        my_template.append(temp)
    return my_template

def make_result_file(dataset_path,output_path_result,output_path_probability,classification_report_file_path,variable,texts,out_path):
    my_template=make_template(texts)
    dataset1,list_of_dict1=load_dataset(input_datafile_name=dataset_path)
    accuracy_val,f1_score_val,truth_val,prob_dict_val=find_accuracy(templeate=my_template,list_of_dict=list_of_dict1,dataset_to_pass=dataset1["test"],texts=texts,classification_report_file_path=classification_report_file_path,variable=variable,out_path=out_path)
    dataframe_res,dataframe_prob=make_csv(accuracy_=accuracy_val,f1_score_=f1_score_val,truth=truth_val,prob_dict_=prob_dict_val,output_csv_result=output_path_result,output_csv_probability_score=output_path_probability)
    return dataframe_res,dataframe_prob

def main(args):
    DATASET_PATH = args.dataset_path
    PROB_PATH = args.output_prob_path
    RESULT_PATH = args.output_result_path
    REPORT_FILE_PATH = args.report_file_path
    PRED_FILE_PATH = args.prediction_fl_pth
    TEMP_PATH = args.temp_fl_pth
    VAR = args.variable
    

    #load template
    texts_nw=pkl.load(open(TEMP_PATH,"rb"))
    dataframe=make_result_file(dataset_path=DATASET_PATH,output_path_result=RESULT_PATH,output_path_probability=PROB_PATH,classification_report_file_path=REPORT_FILE_PATH,variable=VAR,texts=texts_nw,out_path=PRED_FILE_PATH)



if __name__ == "__main__":
  parser= argparse.ArgumentParser(description="ICL-WK")

  parser.add_argument("--dataset_path",type=str,default='',
                      help="Give the dataset path .csv format properly")

  parser.add_argument("--output_prob_path",
                      type=str,default='', help="Give the  path to store probability scores")

  parser.add_argument("--output_result_path",
                        type=str,default='', help="Give the  path to store result file")

  parser.add_argument("--report_file_path",
                          type=str,default='', help="Give the  path to store classification report file")

  parser.add_argument("--prediction_fl_pth",
                            type=str,default='', help="Give the  path to store prediction pickle file")

  parser.add_argument("--temp_fl_pth",
                              type=str,default='', help="Give the  path to store prediction pickle file")

  parser.add_argument("--variable",
                      type=int,
                      default = 0,
                      help="Give 0 for zero-shot",
                      )

#   parser.add_argument("--batch_size",type=int,
#                       default = 128,
#                       help = "Number of data per batch. Default is 16")



  args = parser.parse_args()

  print(args)
  main(args)
