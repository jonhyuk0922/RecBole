# @Time   : 2022/8/29
# @Author : Jo_nyuk
# @Email  : jonhyuk0922@naver.com

from recbole.quick_start import load_data_and_model
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from recbole.utils.case_study import full_sort_topk
#### ----------------- * ----------------- * ----------------- * ####

interaction = pd.read_csv('/Users/jo_nyuk/Desktop/Dev/Recbole/RecBole/dataset/bibly/bibly.inter',sep='/t')
# user_id:token |item_id:token |timestamp:float | target:float (다 0인 상태)

model_path = '/Users/jo_nyuk/Desktop/Dev/Recbole/RecBole/saved/BPR-Aug-26-2022_14-40-01.pth'

# model, dataset 불러오기
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_path)
print(model)
# device 설정
device = config.final_config_dict['device']
print("device: ",device)

# user , item id -> token 변환
user_id2token = dataset.field2id_token['user_id']
item_id2token = dataset.field2id_token['item_id']

# user-item sparse matrix
matrix = dataset.inter_matrix(form='csr')

# user_id , predict item id 저장 변수
pred_list = None
user_list = None

model.eval()
cnt = 0
for data in test_data:
    cnt += 1
    interaction = data[0].to(device)
    score = model.full_sort_predict(interaction)

    rating_pred = score.cpu().data.numpy().copy()
    batch_user_index = interaction['user_id'].cpu().numpy()
    rating_pred[(matrix[batch_user_index].toarray() > 0).squeeze()] = 0

    ind = np.argpartition(rating_pred, -30)[-30:]
    arr_ind = rating_pred[ind]
    arr_ind_argsort = np.argsort(arr_ind)[::-1]

    batch_pred_list = ind[arr_ind_argsort]

    # 예측값 저장
    if pred_list is None:
        pred_list = batch_pred_list
        user_list = batch_user_index
    else:
        pred_list = np.append(pred_list, batch_pred_list, axis=0)
        user_list = np.append(user_list, batch_user_index, axis=0)

result = []

for idx in range(len(user_list)):
    user = user_list[idx]
    items = pred_list[idx*10:(idx+1)*10]
    
    result.append((int(user_id2token[user]), [int(item_id2token[pred]) for pred in items]))

# 데이터 저장
dataframe = pd.DataFrame(result, columns=["user", "item"])
dataframe.sort_values(by='user', inplace=True)
dataframe.to_csv(
    "./saved/recommend/submission.csv", index=False
)
print('inference done!')
