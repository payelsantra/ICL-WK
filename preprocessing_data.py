from tqdm import tqdm
import json

def make_data(given_path):
    with open( given_path, 'r') as json_file:
        json_list = list(json_file)
    wh_id=[]
    wh_label=[]
    wh_clm=[]
    for json_str in tqdm(json_list):
      result = json.loads(json_str)
      d_list=result["id"]
      l=result["label"]
      clm=result['claim']
      if l=="NOT ENOUGH INFO":
        continue
      else:
        wh_id.append(d_list)
        wh_label.append(l)
        wh_clm.append(clm)
    for i in range(len(wh_label)):
        if wh_label[i] == 'SUPPORTS':
            wh_label[i] = 1
        if wh_label[i] == 'REFUTES':
            wh_label[i] = 0
    return wh_id,wh_label,wh_clm

wh_id_tr,wh_label_tr,wh_clm_tr=make_data(given_path="fever/train.jsonl")
wh_id,wh_label,wh_clm=make_data(given_path="fever/shared_task_dev.jsonl")
df = pd.DataFrame(list(zip(wh_id, wh_label,wh_clm)),columns=["ID","LABEL","CLAIM"])
df.to_csv('unique_claim_test_set.csv')
df_tr = pd.DataFrame(list(zip(wh_id_tr,wh_label_tr,wh_clm_tr)),columns=["ID","LABEL","CLAIM"])
df_tr.to_csv('unique_claim_train_set.csv')
