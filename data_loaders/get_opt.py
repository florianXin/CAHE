import os
from argparse import Namespace
import re
from os.path import join as pjoin


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    if str(numStr).isdigit():
        flag = True
    return flag

def get_opt(opt_path, device):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = bool(value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    opt_dict['which_epoch'] = 'latest'

    if opt.dataset_name == "imgToFix": # stage 1: Gaze Fixation Generation Stage
        opt.input_dir = pjoin(opt.data_root, 'dataset/fix_data_stage1')
        opt.cond_dir = pjoin(opt.data_root, 'dataset/image_data_stage1')
    elif opt.dataset_name == "fixToGaze": # stage 2: Head-and-Eye Motion Generation Stage
        opt.input_dir = pjoin(opt.data_root, 'dataset/gazeAndHeadRot_stage2')
        opt.cond_dir = pjoin(opt.data_root, 'dataset/fixAndHeadPos_stage2')
    
    opt.is_train = False
    opt.device = device

    return opt