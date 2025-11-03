import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import torch.nn.functional as F
import torch.nn as nn
import copy
import random
import torch
from torch.optim.lr_scheduler import _LRScheduler
import scipy
from sklearn.metrics import r2_score,mean_squared_error

class WarmupThenReduceLROnPlateau(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, mode='min', factor=0.1, patience=10, min_lr=0):
        self.warmup_epochs = warmup_epochs
        self.scheduler_after_warmup = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, factor, patience, min_lr)
        super(WarmupThenReduceLROnPlateau, self).__init__(optimizer)

    def get_lr(self, metrics=None):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            if self.last_epoch == self.warmup_epochs:
                self.scheduler_after_warmup.step(100)
            return self.scheduler_after_warmup.get_last_lr()

    def step(self, metrics=None):
        if self.last_epoch >= self.warmup_epochs:
            self.scheduler_after_warmup.step(metrics)
            # print(self.scheduler_after_warmup.best)
        super(WarmupThenReduceLROnPlateau, self).step()

class ModelTrainer:
    def __init__(self, model, init_lr=0.01,mode = 'delta_est'):
        self.model = model.cuda()

        # self.model.apply(self.init_weights)

        self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                           lr=init_lr,
                                           weight_decay=0.01)
        # self.scheduler = WarmupThenReduceLROnPlateau(self.optimizer,
        #                                              warmup_epochs=10,
        #                                              mode='min',
        #                                              factor=0.1,
        #                                              patience=100,
        #                                              min_lr=1e-7)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                    mode='min', 
                                                                    factor=0.5, 
                                                                    patience=100, 
                                                                    min_lr=1e-7)
        self.mode = mode

        self.ckpt_dir = f'ckpt/best_model_{self.mode}.pth'

        # self.loss = nn.MSELoss(reduction='mean')
        self.loss = nn.L1Loss(reduction='mean')
        # self.loss = nn.SmoothL1Loss(reduction='mean')

    def init_weights(self,m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def train(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for inputs in tqdm(dataloader):

            self.optimizer.zero_grad()

            input = inputs['ir'].float().cuda()
            # noise = np.random.normal(0, 0.05, input.shape)
            # noise = torch.from_numpy(noise).float().cuda()
            # input = input + noise
            # # input = np.clip(input,0,1)
            # input = torch.clamp(input, 0, 1)
            label = inputs[self.mode].float().cuda().unsqueeze(1)
            output= self.model(input)
            loss = self.loss(output,label) 

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()        

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs in tqdm(dataloader):
                input = inputs['ir'].float().cuda()
                label = inputs[self.mode].float().cuda().unsqueeze(1)
                output= self.model(input)
                loss = self.loss(output,label) 

                # loss = self.loss1(ori_data,pred_data) + self.loss2(ori_mask.squeeze(),pred_mask.squeeze(),self.tgt)
                
                total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def test(self, dataloader):
        # self.model.emb.load_state_dict(torch.load(f'ckpt/{self.mode}/emb_best_model.pth'))  # PATH是你的模型文件路径
        # self.model.encoder.load_state_dict(torch.load(f'ckpt/{self.mode}/encoder_best_model.pth'))  
        load_ckpt = torch.load(self.ckpt_dir,weights_only=False)                                 # 从断点路径加载断点，指定加载到CPU内存或GPU
# 简单验证
        load_weights_dict = {k: v for k, v in load_ckpt['parameter'].items()
                                      if self.model.state_dict()[k].numel() == v.numel()}  # 简单验证
        self.model.load_state_dict(load_weights_dict, strict=False)
        self.model.eval()
        total_loss = 0.0
        draw_datas = []
        with torch.no_grad():
            for inputs in tqdm(dataloader):
                input = inputs['ir'].float().cuda()
                label = inputs[self.mode].float().cuda().unsqueeze(1)
                output= self.model(input)
                loss = self.loss(output,label) 

                draw_datas.append((float(output),float(label)))

                total_loss += loss.item()

        return total_loss / len(dataloader),draw_datas




    def start_training(self, train_dataloader, valid_dataloader,test_loader, epochs,continue_train=False):
        best_loss = float('inf')
        try:
            if continue_train:
                load_ckpt = torch.load(self.ckpt_dir,weights_only=False)                                 # 从断点路径加载断点，指定加载到CPU内存或GPU
                load_weights_dict = {k: v for k, v in load_ckpt['parameter'].items()
                                        if self.model.state_dict()[k].numel() == v.numel()}  # 简单验证
                self.model.load_state_dict(load_weights_dict, strict=False)
                best_loss = float(load_ckpt['best_loss'])

                self.optimizer.load_state_dict(load_ckpt['optimizer'])
            
            for epoch in range(epochs):

                train_loss = self.train(train_dataloader)
                valid_loss = self.evaluate(valid_dataloader)


                self.scheduler.step(valid_loss)
                save_log = ''
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    checkpoint = {
                                'parameter': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'best_loss': best_loss}

                    torch.save(checkpoint,self.ckpt_dir)
                    save_log = ', save best model!'
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Current LR: {self.scheduler.get_last_lr()}{save_log}')
        except KeyboardInterrupt:
            print('Stop Training, Draw Picture Now.')
        test_loss,draw_datas = self.test(test_loader)

        print(f'Test Loss: {test_loss:.4f}')

        self.draw_pict(draw_datas)


    def draw_pict(self,draw_datas):

        titles = ['$\delta_{est}$','$SOC_{t_1-s_1}$','$SOC_{t_1-s_0}$','$FOSC_{s_1}$','HOMO/LUMO Gap','pV$_{esp}$','$Wave Number$','$E_{t_1}$','$E_{t_1}(vertical)$','$E_{s_1}$']
        units = ['$(eV)$','$(cm^{-1})$','$(cm^{-1})$','','(a. u.)','(a. u.)$^2$','$(cm^{-1})$','$(eV)$','$(eV)$','$(eV)$']
        units_bra = ['$eV$','$cm^{-1}$','$cm^{-1}$','','(a. u.)','(a. u.)$^2$','$cm^{-1}$','$eV$','$eV$','$eV$']
        colors = ['#d8f3dc', '#b7e4c7', '#95d5b2', '#74c69d', '#52b788', '#40916c', '#2d6a4f', '#1b4332']

        if self.mode == 'delta_est':
            n = 0
        elif self.mode == 'fosc_s1':
            n = 3
        elif self.mode =='soc_t1':
            n = 1
        elif self.mode =='soc_t1s0':
            n = 2
        elif self.mode =='homo_lumo_gap':
            n = 4
        elif self.mode =='p_log_esp':
            n = 5
        else:
            n = 0


        preds = []
        trues = []
        for item in draw_datas:
            preds.append(item[0])
            trues.append(item[1])

        with open(f'figs/{self.mode}.txt','w') as f:
                f.write('True,Pred\n')
                for i in range(len(preds)):
                    f.write(f'{trues[i]},{preds[i]}\n')
        plt.figure(figsize=(8, 8))
        plt.scatter(trues, preds, color=colors[n], s=10,alpha=0.9)
        min_val, max_val = np.min([np.min(trues), np.min(preds)]), np.max([np.max(trues), np.max(preds)])
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        r2 = r2_score(trues,preds)
        rmse = np.sqrt(mean_squared_error(trues, preds))
        plt.text(min_val, max_val*0.98, 
                 f'$r^2 = {r2:.4f}$\n$RMSE = {rmse:.4f}${units_bra[n]} ', 
                 fontsize=18, style='italic',
                 ha='left', va='top')
        plt.xlabel(f'Calculated {titles[n]} {units[n]}', fontsize=18, fontstyle='italic')
        plt.ylabel(f'Predicted {titles[n]} {units[n]}', fontsize=18, fontstyle='italic')
        # plt.title(f'R2:{round(r2,4)}')
        plt.title(titles[n], fontsize=20, fontstyle='italic')

        plt.savefig(f'figs/{self.mode}.png')
        plt.savefig(f'figs/{self.mode}.svg',transparent=True)

        plt.cla()

