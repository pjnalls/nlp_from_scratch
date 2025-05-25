from io import open
import os
import time


import torch
from torch.utils.data import Dataset

import src.common.tools as tools


class NamesDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_time = time.localtime
        labels_set = set()

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        # Read all the ``.txt`` files in the specified directory
        text_files = [x for x in os.listdir(os.getcwd() 
            if 'nlp_from_scratch' in os.getcwd() 
            else os.getcwd() + '/nlp_from_scratch' + data_dir) 
            if x.endswith('.txt')]
        text_files.sort()
        # print(text_files)
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)
            lines = open(os.getcwd() + data_dir +
                         f"/{filename}", encoding='utf-8').read().strip().split('\n')
            for name in lines:
                self.data.append(name)
                self.data_tensors.append(tools.line_to_tensor(name))
                self.labels.append(label)

        # Cache the tensor representation of the labels
        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor(
                [self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]

        return label_tensor, data_tensor, data_label, data_item


alldata = NamesDataset("/data/names")
# print(f"loaded {len(alldata)} items of data.")
# print(f"example item: {alldata[0]}")
train_set, test_set = torch.utils.data.random_split(alldata, [.85, .15])
# generator = torch.Generator(device=tools.device).manual_seed(2024)
# print(f"train examples: {len(train_set)}")
# print(f"test examples: {len(test_set)}")
