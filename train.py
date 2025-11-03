import torch.utils
import torch.utils.data
from data.IRDataSet import IRDataSet
from data.QM9Set import QM9IRDataSet
from data.ConSet import Set
from model.CNN import SpectrumCNN
from data.ESPSet import ESPSet
from model.MLP import ResidualMLP
from model.TSFM import DeepSpectralTransformer
from model.ConvAttn import ConvInputSpectralTransformer
from model.ResCNN import DeepIRConvNet
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset,Dataset
from model.trainner import ModelTrainer
import argparse
parser = argparse.ArgumentParser(description='IR spectrum prediction')
# parser.add_argument('--mode', type=str, default='esp2', help='a')
parser.add_argument('--mode', type=str, default='delta_est', help='a')
# parser.add_argument('--mode', type=str, default='homo_lumo_gap', help='a')
parser.add_argument('--bsz', default=1024, type=int, help='batchsize')
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# dataset = IRDataSet("data/common_rows_with_gap.csv")
# dataset = ESPSet('data/merged_file.csv')
dataset = QM9IRDataSet("/home/yanggk/Data/SpecGLU/QM9/SpecBert/gaussian_summary.csv")



set_size = len(dataset)
idx = np.arange(0,set_size)
np.random.shuffle(idx)


print(set_size)
train_size = int(0.8*set_size)
test_size = int(0.1*set_size)
val_size = set_size - train_size - test_size


train_idx = idx[:train_size]
test_idx = idx[train_size:test_size+train_size]
val_idx = idx[test_size+train_size:]

np.save(f'data/train.npy',train_idx)
np.save(f'data/test.npy',test_idx)
np.save(f'data/valid.npy',val_idx)

train_idx = np.load(f'data/train.npy') 
val_idx = np.load(f'data/valid.npy')
test_idx = np.load(f'data/test.npy')

train_dataset = Subset(dataset,train_idx)
augment_types = ['shift', 'scale', 'quant']
train_dataset = list(train_dataset)  # 转换为列表以便扩展
original_train_size = len(train_dataset)

# 对每种增强类型扩展训练集
for aug_type in augment_types:
    augmented_data = []
    for entry in train_dataset[:original_train_size]:  # 遍历原始训练集
        new_entry = entry.copy()
        new_entry['ir'] = dataset.apply_augmentation(new_entry['ir'], aug_type)
        augmented_data.append(new_entry)
    train_dataset.extend(augmented_data) 

validate_dataset = Subset(dataset,val_idx)
test_dataset = Subset(dataset,test_idx)

print(f'======train size:{len(train_dataset)}======')
print(f'======valid size:{len(validate_dataset)}======')
print(f'======test size:{len(test_dataset)}======')

train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, num_workers=4,pin_memory=True)
validate_loader = DataLoader(validate_dataset, batch_size=args.bsz, shuffle=True, num_workers=4,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

# model = DeepIRConvNet()
model = SpectrumCNN()

trainer = ModelTrainer(model,0.0002,mode=args.mode)

trainer.start_training(train_dataloader=train_loader,
                       valid_dataloader=validate_loader,
                       test_loader=test_loader,   
                       epochs=1000,
                       continue_train=False)

