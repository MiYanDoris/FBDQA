import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils import data

def get_raw_data(days):
    '''
        Output:
        train_data, train_label, val_data, val_label, test_data, test_label
        data: N,100,26
        label: N,5
    '''
    file_dir = "/home/hewang/ym/LOB/demo/FBDQA2021A_MMP_Challenge_ver0.2/data"

    df = pd.DataFrame()
    for sym in range(10):
        for date in range(days):
            if (date & 1):
                file_name = f"snapshot_sym{sym}_date{date//2}_am.csv"
            else:
                file_name = f"snapshot_sym{sym}_date{date//2}_pm.csv"
            if not os.path.isfile(os.path.join(file_dir,file_name)):
                print('No file', file_name)
                continue
            new_df = pd.read_csv(os.path.join(file_dir,file_name))
            df = df.append(new_df)
        
    label_col_name = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']

    n = len(df)
    ##划分训练/测试集
    train_nums = int(n*0.8)
    val_nums = int(n*0.1)
    train_data = np.ascontiguousarray(df.iloc[:, 2:-5][:train_nums].values)
    train_label = df[label_col_name][:train_nums].values.reshape(-1, 5)

    val_data = np.ascontiguousarray(df.iloc[:, 2:-5][train_nums:train_nums+val_nums].values)
    val_label = df[label_col_name][train_nums:train_nums+val_nums].values.reshape(-1, 5)

    test_data = np.ascontiguousarray(df.iloc[:, 2:-5][train_nums+val_nums:].values)
    test_label = df[label_col_name][train_nums+val_nums:].values.reshape(-1, 5)
    return train_data, train_label, val_data, val_label, test_data, test_label

def data_preprocess(x, device):
    '''
        TODO: 
            whether to use sym???
            how to scale up price: raw 1e-2, I currently multiply it by 100 and divide amount by 10

        x: 1, 100, 24: sym	n_close	amount_delta n_midprice bid_i bsize_i ask_i asize_i
            n_bid_i: 4+2i
            bsize_i 5+2i
            aski: 14+2i
            n_asize_i: 15+2i

        output: 1, 100, 31
    '''
    input = torch.zeros((1, 100, 32))

    # bid, ask, size
    for i in range(10):
        input[..., 2*i] = 100 * x[..., 4+2*i]
        input[..., 2*i+1] = torch.log(x[..., 5+2*i] + 1e-10) / 10
    
    # spread & mid_price & weighted_price
    for i in range(3):
        input[...,20+2*i] = 100 * (x[..., 14+2*i] - x[..., 4+2*i])
        input[...,21+2*i] = 100 * (x[..., 14+2*i] + x[..., 4+2*i]) / 2
        input[...,26+i] = 100 * (x[..., 14+2*i] * x[..., 5+2*i] + x[..., 4+2*i] * x[..., 15+2*i]) / (x[..., 5+2*i] + x[..., 15+2*i])
    
    input[..., 29] = torch.log1p(x[..., 2]) / 10 # amount
    input[..., 30] = (x[..., 5] - x[..., 15]) / (x[..., 5] + x[..., 15])

    bsize_sum = torch.sum(x[..., 5:15:2], axis=-1)
    asize_sum = torch.sum(x[..., 15:25:2], axis=-1)
    input[..., 31] = (bsize_sum - asize_sum)/(bsize_sum + asize_sum)
    
    return input.to(device)

def data_transform(X, T):
    [N, D] = X.shape
    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = X[i - T:i, :]
    return dataX

class Dataset(data.Dataset):
    def __init__(self, data, label, T, device, max_len):
        self.T = T
        data = data_transform(data, self.T)
        self.x = torch.tensor(data).to(torch.float32).unsqueeze(1)
        self.y = torch.tensor(label[T - 1:].astype(np.int64))
        if len(self.x) > max_len:
            self.x = self.x[:max_len]
            self.y = self.y[:max_len]
        self.length = len(self.x)
        self.device=device

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_input = self.x[index]
        return data_preprocess(raw_input, self.device), self.y[index].to(self.device)

def get_dataloader(period, device, batch_size, days):
    '''
        period: tick5, 10, ...
        x: 1, 100, 24
        y: 5
    '''
    train_data, train_label, val_data, val_label, test_data, test_label = get_raw_data(days)

    dataset_train = Dataset(data=train_data,label=train_label[:, period], T=100, device=device, max_len=5000000)
    dataset_val   = Dataset(data=val_data,  label=val_label[:, period],   T=100, device=device, max_len=50000)
    dataset_test  = Dataset(data=test_data, label=test_label[:, period],  T=100, device=device, max_len=50000)
    print('train dataset: ', len(dataset_train))
    print('val dataset: ', len(dataset_val))
    print('test dataset: ', len(dataset_test))

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_data, train_label, val_data, val_label, test_data, test_label = get_raw_data(30)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_train = Dataset(data=train_data,label=train_label, T=100, device=device)

    print(dataset_train[10][0].shape)
    print(dataset_train[10][1].shape)
    print(dataset_train[10][0])
    print(dataset_train[10][1])
    train_loader, val_loader, test_loader = get_dataloader(device, batch_size=512, days=30)
    print(len(train_loader))

