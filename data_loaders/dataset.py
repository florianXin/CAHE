import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import codecs as cs
from tqdm import tqdm

from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

from data_loaders.get_opt import get_opt
class CAHEDataset(data.Dataset):
    def __init__(self, datapath='./dataset/imgToFix_opt.txt', split="train", **kwargs):
    # def __init__(self, datapath='./dataset/fixToGaze_opt.txt', split="train", **kwargs):

        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None 
        opt = get_opt(dataset_opt_path, device)
        opt.input_dir = pjoin(abs_base_path, opt.input_dir)
        opt.cond_dir = pjoin(abs_base_path, opt.cond_dir)

        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.data_root = pjoin(opt.data_root, "dataset")

        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        if opt.dataset_name == "imgToFix":
            self.mean = np.load(pjoin(opt.data_root, 'inputStandardization/mean_s1.npy'))
            self.std = np.load(pjoin(opt.data_root, 'inputStandardization/std_s1.npy'))
        elif opt.dataset_name == "fixToGaze":
            self.mean = np.load(pjoin(opt.data_root, 'inputStandardization/mean_s2.npy'))
            self.std = np.load(pjoin(opt.data_root, 'inputStandardization/std_s2.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        self.dataset = Dataset(self.opt, self.mean, self.std, self.split_file)

        assert len(self.dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.dataset.__getitem__(item)

    def __len__(self):
        return self.dataset.__len__()


class Dataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                input = np.load(pjoin(opt.input_dir, name + '.npy'))
                if (len(input)) < 1:
                    continue
                cond = np.load(pjoin(opt.cond_dir, name + '.npy'), )
                if (len(cond)) < 1:
                    continue
                data_dict[name] = {'input': input,
                                    'length': len(input),
                                    'cond': cond}
                new_name_list.append(name)
                length_list.append(len(input))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        input, m_length, cond_list = data['input'], data['length'], data['cond']
        
        input = (input - self.mean) / self.std

        # Semantic Feature Ablation
        # cond_list = np.delete(cond_list, [2,9,10], axis=1)
        
        # Dynamic Feature Ablation
        # cond_list = np.delete(cond_list, [3,4,5,6,7,8], axis=1)
 
        return input, m_length, cond_list