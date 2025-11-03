from data.IRDataSet import IRDataSet
from data.QM9Set import QM9IRDataSet,QM9IRESPDataSet
from model.CNN import SpectrumCNN
from data.ESPSet import ESPSet
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='IR spectrum prediction')
parser.add_argument('--mode', type=str, default='delta_est', help='a')
parser.add_argument('--bsz', default=1024, type=int, help='batchsize')
args = parser.parse_args()



# dataset = IRDataSet("data/common_rows_with_gap.csv")
dataset = ESPSet("/home/yanggk/Data/SpecGLU/ESP/indices.csv")
# dataset = QM9IRDataSet("/home/yanggk/Data/SpecGLU/QM9/SpecBert/gaussian_summary.csv")
# dataset = QM9IRESPDataSet("/home/yanggk/Data/SpecGLU/QM9/SpecBert/info_smi.csv")

index = np.random.choice(len(dataset), size=200, replace=False)
subset = torch.utils.data.Subset(dataset, index)

data_loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=4,pin_memory=True)

model = SpectrumCNN()

load_ckpt = torch.load(f'ckpt/best_model_{args.mode}.pth',weights_only=False)                                 # 从断点路径加载断点，指定加载到CPU内存或GPU
load_weights_dict = {k: v for k, v in load_ckpt['parameter'].items()
                                        if model.state_dict()[k].numel() == v.numel()}  # 简单验证
model.load_state_dict(load_weights_dict, strict=False)


# model.load_state_dict(torch.load(f'ckpt/best_model_{args.mode}.pth'))
model.eval()

preds = []
smiles = []
trues = []
for data in tqdm(data_loader):
    pred = model(data['ir'].float().squeeze(1))
    true = data[args.mode]
    smi = data['smiles']
    preds.append(float(pred))
    trues.append(float(true))
    smiles.append(str(smi))

with open(f'data/trans_{args.mode}.csv','w') as f:
    f.write('smi,True,Pred\n')
    for i in range(len(preds)):
        f.write(f'{smiles[i]},{trues[i]},{preds[i]}\n')




