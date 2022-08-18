import torch
import torch.nn as nn

from exceptions.exceptions import InvalidModelError
from torch.nn import LSTM, Linear, GRU, RNN


class BaseRNN(nn.Module):
    def __init__(self, num_sensors, hidden_size, sequence_length, dropout, batch_size, 
                 device, output_size, num_layers=1, arch=None):

        super(BaseRNN,self).__init__()
        self.num_sensors = num_sensors         # num features
        self.hidden_size = hidden_size         # size of hidden state vecot
        self.sequence_length = sequence_length # length of sequence to parse
        self.dropout = nn.Dropout(dropout)     # dropout   
        self.batch_size = batch_size           # batch size 
        self.device = device                   # gpu id
        self.output_size = output_size         # num classes
        self.num_layers = num_layers           # no stacking 
        self.model_name = arch                 # model to use: lstm, gru, or rnn

        self.model_dict = {"lstm": LSTM, "gru": GRU, "rnn": RNN}
        self.model = self._get_basemodel(self.model_name, self.num_sensors)

        self.fc = Linear(self.hidden_size, self.output_size)

    def _get_basemodel(self, model_name, input_size):
        try: 
            model = self.model_dict[model_name]
        except KeyError:
            raise InvalidModelError(
                "Invalid model architecture selected. Pass one of: lstm, rnn, gru"
            )
        else:
            return model(input_size = input_size,
                         hidden_size = self.hidden_size,
                         batch_first = True, 
                         num_layers = self.num_layers)

    def forward(self, x):
        if self.model_name == 'lstm': 
            hidden = (torch.zeros((self.num_layers, self.batch_size, self.hidden_size), device=torch.device('cuda')),
                      torch.zeros((self.num_layers, self.batch_size, self.hidden_size), device=torch.device('cuda')))
        else: 
            hidden = (torch.zeros((self.num_layers, self.batch_size, self.hidden_size), device=torch.device('cuda')))
        
        model_out,hidden = self.model(x, hidden)
        out = self.dropout(model_out)
        out = self.fc(out[:,-1])

        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, momentum=0.99, 
                 epsilon=0.001, squeeze=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
        self.relu = nn.ReLU()
        self.zp = nn.ConstantPad1d(((kernel_size-1), 0), 0)

    def forward(self, x):
        x = self.zp(x)
        x = self.conv(x)
        # input (batch_size, out_channel, L_out)
        x = self.bn(x)
        # same shape as input
        y = self.relu(x)
        return y

class CRNN(nn.Module):
    def __init__(self, num_features, output_size, hidden_size, kernel_size1=2, kernel_size2=2, 
                 num_layers=2, conv1_nf=64, conv2_nf=128, dropout=0.1):
        
        super(CRNN,self).__init__()

        self.conv1 = ConvBlock(num_features, conv1_nf, kernel_size1)
        self.conv2 = ConvBlock(conv1_nf, conv2_nf, kernel_size2)
        self.gap = nn.AvgPool1d(kernel_size=output_size)
        self.lstm = LSTM(input_size=conv2_nf, 
                         hidden_size=hidden_size,
                         num_layers= num_layers,
                         batch_first=True) 
        self.dropout = nn.Dropout(dropout)
        self.fc= Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0,2,1) # input has to be (batch_size, num_features, seq_length)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.gap(out)
        
        out = out.permute(0,2,1)
        model_out, _ = self.lstm(out)
        out = self.dropout(model_out)
        out = self.fc(out[:,-1])

        return out

class LSTMfcn(nn.Module):
    def __init__(self, num_features, output_size, hidden_size, kernel_size1=8, kernel_size2=5, 
                 kernel_size3=3,num_layers=1, conv1_nf=128, conv2_nf=256, conv3_nf=128, dropout=0.1):
        super(LSTMfcn,self).__init__()

        self.conv1 = ConvBlock(num_features, conv1_nf, kernel_size1)
        self.conv2 = ConvBlock(conv1_nf, conv2_nf, kernel_size2)
        self.conv3 = ConvBlock(conv2_nf, conv3_nf, kernel_size3)
        self.gap = nn.AvgPool1d(kernel_size=output_size)
        self.lstm = nn.LSTM(input_size=num_features, 
                            hidden_size=conv3_nf,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(conv3_nf * 2, output_size)

    def forward(self, x):
        # branch 1
        out1, (ht,ct) = self.lstm(x)
        out1 = out1[:,-1]
        # branch 2
        out2 = x.permute(0,2,1)
        out2 = self.conv1(out2)
        out2 = self.conv2(out2)
        out2 = self.conv3(out2)
        out2 = self.gap(out2)
        out2 = torch.mean(out2,2)
        # concat
        x_all = torch.cat([out1, out2], dim=1)
        x_out = self.fc(x_all)

        return x_out 

    
