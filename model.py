import torch.nn as nn
import torch
from torchinfo import summary

class DeepLOB(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 3
        
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,1),stride=(2,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1),stride=(2,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,8)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1),stride=(2,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
       
        # lstm layers
        self.fc = nn.Sequential(nn.Linear(384, 64),nn.Linear(64, self.num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        x = x.reshape(-1,48*8)
        x = self.fc(x)

        forecast_y = torch.softmax(x, dim=1)

        return forecast_y

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(3200,128),
                    nn.LeakyReLU(),
                    nn.Linear(128,64),
                    nn.LeakyReLU(),
                    nn.Linear(64,64),
                    nn.LeakyReLU(),
                    nn.Linear(64,3)
                )
        
    def forward(self,x):
        x = x.view(-1,100*32)
        x = self.net(x)
        return torch.softmax(x, dim=1)

class attn_module(nn.Module):
    def __init__(self, d_model=128, no_linear=False, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.no_linear = no_linear
        if not no_linear:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(d_model)
            self.activation = nn.LeakyReLU()

    def forward(self, src1_ori, pos1_ori, src2_ori, pos2_ori, attn=True):
        '''
        :param src1: q [B, C, N]
        :param pos1: positional embedding[B, C, N]
        :param src2: k, v [B, C, M]
        :param pos2: positional embedding[B, C, M]
        :return: [B, C, N]
        '''
        src1 = src1_ori.permute(2, 0, 1)
        pos1 = pos1_ori.permute(2, 0, 1)
        src2 = src2_ori.permute(2, 0, 1)
        pos2 = pos2_ori.permute(2, 0, 1)

        src12, weight = self.attn(src1 + pos1, src2 + pos2,
                             value=src2)
        src1_new = src1 + self.dropout1(src12)
        if not attn:
            src1_new = src1
        src1_new = self.norm1(src1_new)
        if not self.no_linear:
            src13 = self.linear2(self.dropout2(self.activation(self.linear1(src1_new))))
            src1_new = src1_new + self.dropout3(src13)
            src1_new = self.norm2(src1_new)
        return src1_new.permute(1,2,0)

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer,self).__init__()
        self.embed = nn.Sequential(
                    nn.Conv1d(32, 64, kernel_size=1),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU()
                )
        self.transformer_1 = attn_module(128)
        self.linear_1 = nn.Conv1d(256, 128, 1)
        self.transformer_2 = attn_module(128)
        self.linear_2 = nn.Conv1d(128, 32, 1)
        self.classifier = nn.Sequential(
                    nn.Linear(800,128),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Linear(128,64),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU(),
                    nn.Linear(64,64),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU(),
                    nn.Linear(64,3)
                )

        self.pos_emb = nn.Linear(1, 128)
        
    def forward(self,x):
        x = x.squeeze().transpose(-1, -2).contiguous() #B,C,100
        B, _, _ = x.shape
        pos = torch.arange(50, device=x.device, dtype=torch.float).unsqueeze(-1) #50,1
        pos_emb = self.pos_emb(pos) # 50,128
        pos_emb = pos_emb.unsqueeze(0).repeat(B,1,1).transpose(-1, -2).contiguous() #B,128,50
        
        embed_0 = self.embed(x) # B,64,100
        embed_0 = embed_0.transpose(-1, -2).contiguous() # B,100,64
        two_ticks = embed_0.reshape(B, 50, 2 * 64).transpose(-1, -2).contiguous() # B,2C,50
        feature_1 = self.transformer_1(two_ticks, pos_emb, two_ticks, pos_emb) # B,128,50
        feature_1 = feature_1.transpose(-1, -2).reshape(B, 25, 2 * 128).transpose(-1, -2).contiguous() # B,25,256
        embed_1 = self.linear_1(feature_1) # B,128,25
        feature_2 = self.transformer_1(embed_1, pos_emb[..., :25], embed_1, pos_emb[..., :25]) # B,128,25
        global_feature = self.linear_2(feature_2).reshape(B, -1) # B,800
        classification = self.classifier(global_feature)

        return torch.softmax(classification, dim=1)

class deeplob(nn.Module):
    def __init__(self, y_len=3):
        super().__init__()
        self.y_len = y_len
        
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,8)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        
        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros((1, x.size(0), 64), device=x.device)
        c0 = torch.zeros((1, x.size(0), 64), device=x.device)
    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        print(x_inp1.shape, x_inp2.shape, x_inp3.shape)  
        
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)
        
        return forecast_y

class deeplob_bn(nn.Module):
    def __init__(self, y_len=3):
        super().__init__()
        self.y_len = y_len
        
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.BatchNorm2d(32),
            nn.Tanh(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,8)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
        )
        
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
        )
        
        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros((1, x.size(0), 64), device=x.device)
        c0 = torch.zeros((1, x.size(0), 64), device=x.device)
    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        print(x_inp1.shape, x_inp2.shape, x_inp3.shape)  
        
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)
        
        return forecast_y

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = deeplob_bn(3)
    model.to(device)
    summary(model, (2, 1, 100, 32))
