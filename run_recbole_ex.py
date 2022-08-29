# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn


import argparse
from recbole.quick_start import run_recbole


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='bibly', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--config_dict', type=str, default=None, help='config_dict')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    # RecBole/recbole/quick_start/quick_start.py 에 있는 run_recbole 
    run_recbole(model=args.model, dataset=args.dataset, config_dict={"data_path":"/Users/jo_nyuk/Desktop/Dev/Recbole/RecBole/dataset/"})
