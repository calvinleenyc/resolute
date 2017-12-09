import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

lr_rate = 0.001

KERNEL_SIZE = 5

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.i2f = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2I = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2O = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2C = nn.Linear(input_size + hidden_size, hidden_size)
        self.H2out = nn.Linear(hidden_size, output_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_rate)
        
    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        forget = F.sigmoid(self.i2f(combined))
        I = F.sigmoid(self.i2I(combined))
        O = F.sigmoid(self.i2O(combined))
        C = forget * cell + I * F.tanh(self.i2C(combined))
        H = O * F.tanh(C)
        output = F.log_softmax(self.H2out(H))
        return output, H, C

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def initCell(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def num_params(self):
        ans = 0
        for param in self.parameters():
            sz = param.size()
            here = 1
            for dim in range(len(sz)):
                here *= sz[dim]
            ans += here
        return ans

class CNN(nn.Module):
    # as described in Appendix B
    def __init__(self, memory):
        super(CNN, self).__init__()
        input_channels = 2 if memory else 1
        
        self.conv1a = nn.Conv2d(input_channels, 8, 1, padding = 0)
        self.conv1b = nn.Conv2d(input_channels, 8, 3, padding = 1)
        self.conv1c = nn.Conv2d(input_channels, 8, 5, padding = 2)
        self.conv1d = nn.Conv2d(input_channels, 8, 7, padding = 3)

        # CAREFUL HERE
        # the text specifies that the batchnorm happens before the concatenation...
        self.batchnormA = nn.BatchNorm2d(8)
        self.batchnormB = nn.BatchNorm2d(8)
        self.batchnormC = nn.BatchNorm2d(8)
        self.batchnormD = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1) # dimension-halving
        self.batchnorm2 = nn.BatchNorm2d(32)
        
    def forward(self, x):
        # input x should have been pre-processed already: saturations in [-1, 1]
        # shape of x is batch_size x input_channels x 28 x 28
        layer1 = F.relu(torch.cat((self.batchnormA(self.conv1a(x)),
                                   self.batchnormB(self.conv1b(x)),
                                   self.batchnormC(self.conv1c(x)),
                                   self.batchnormD(self.conv1d(x))),
                                  dim = 1)
                        )
        layer2 = F.relu(self.batchnorm2(self.conv2(layer1)))

        
        
        
        

class CDNA(nn.Module):
    def __init__(self):
        super(CDNA, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, KERNEL_SIZE, stride = 2, padding = 2)
        self.lstm1 = ConvLSTM(32, 32, 32)
        self.lstm2 = ConvLSTM(32, 32, 32)
        self.downsample23 = nn.Conv2d(32, 64, 2, stride = 2)
        self.lstm3 = ConvLSTM(16, 64, 64)
        self.lstm4 = ConvLSTM(16, 64, 64)
        self.downsample45 = nn.Conv2d(64, 128, 2, stride = 2)
        self.lstm5 = ConvLSTM(8, 138, 138)
        self.to_kernels = nn.Linear(138 * 8 * 8, 10 * 5 * 5)
        self.upsample56 = nn.ConvTranspose2d(138, 64, 2, stride = 2)
        self.lstm6 = ConvLSTM(16, 64, 64)
        self.upsample67 = nn.ConvTranspose2d(64 + 64, 32, 2, stride = 2)
        self.lstm7 = ConvLSTM(32, 32, 32)
        # the end of the diagram is ambiguous
        self.last_upsample = nn.ConvTranspose2d(32 + 32, 32, 2, stride = 2) 
        self.conv2 = nn.Conv2d(32, 11, kernel_size = 1)

        # For some reason, F.softmax(x, dim = 2) doesn't work on my machine,
        # so I use this instead: given a 4D tensor, it softmaxes dimension 1.
        self.softmax = nn.Softmax2d()
        
    def forward(self, img, tiled, hiddens, cells):
        # input is preprocessed with numpy (at least for now)
        layer0 = self.conv1(img)
        hidden1, cell1 = self.lstm1(layer0, hiddens[1], cells[1])
        hidden2, cell2 = self.lstm2(hidden1, hiddens[2], cells[2])
        hidden3, cell3 = self.lstm3(self.downsample23(hidden2), hiddens[3], cells[3])
        hidden4, cell4 = self.lstm4(hidden3, hiddens[4], cells[4])
        
        input5 = torch.cat((self.downsample45(hidden4), tiled), 1)
        hidden5, cell5 = self.lstm5(input5, hiddens[5], cells[5])

        kernels = self.to_kernels(hidden5.view([-1, 138 * 8 * 8])).view([-1, 25, 10, 1])
        # NOT a channel softmax, but a spatial one
        normalized_kernels = torch.transpose(self.softmax(kernels), 1, 2)
        normalized_kernels = torch.stack(torch.split(torch.squeeze(normalized_kernels), 5, dim = 2), dim = -2)
        # We will wait to transform the images until we compute the loss.

        hidden6, cell6 = self.lstm6(self.upsample56(hidden5), hiddens[6], cells[6])
        input7 = self.upsample67(torch.cat((hidden6, hidden3), 1))
        hidden7, cell7 = self.lstm7(input7, hiddens[7], cells[7])

        input_out = self.last_upsample(torch.cat((hidden7, hidden1), 1))
        out = self.softmax(self.conv2(input_out)) # channel softmax

        return out, normalized_kernels, [None, hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7],\
            [None, cell1, cell2, cell3, cell4, cell5, cell6, cell7]

    def initHidden(self, batch_size = 1):
        # The first entry is just so that the indexing aligns with the semantics
        return [None,
                self.lstm1.initHidden(batch_size),
                self.lstm2.initHidden(batch_size),
                self.lstm3.initHidden(batch_size),
                self.lstm4.initHidden(batch_size),
                self.lstm5.initHidden(batch_size),
                self.lstm6.initHidden(batch_size),
                self.lstm7.initHidden(batch_size),
        ]

    def initCell(self, batch_size = 1):
        return [None,
                self.lstm1.initCell(batch_size),
                self.lstm2.initCell(batch_size),
                self.lstm3.initCell(batch_size),
                self.lstm4.initCell(batch_size),
                self.lstm5.initCell(batch_size),
                self.lstm6.initCell(batch_size),
                self.lstm7.initCell(batch_size),
        ]

    def num_params(self):
        ans = 0
        for param in self.parameters():
            sz = param.size()
            here = 1
            for dim in range(len(sz)):
                here *= sz[dim]
            ans += here
        return ans

if __name__ == '__main__':

    # A test for a tricky part of the code
    qe = Variable(torch.FloatTensor(np.random.randn(11, 25, 10, 1)))
    qe = torch.transpose(qe, 1, 2)
    ans1 = qe.contiguous().view([-1, 10, 5, 5])
    ans2 = torch.stack(torch.split(torch.squeeze(qe), 5, dim = 2), dim = -2)
    print(type(ans1))
    print(type(ans2))
    print(F.mse_loss(ans1, ans2))
    #exit(0) # Experiments show that ans2 is slightly faster to compute


    
    rnn = CDNA()
    
    print(rnn.num_params())

    img = np.zeros([3, 64, 64])
    tiled = np.zeros([10, 8, 8])

    img = Variable(torch.FloatTensor([img, img, img, img]))
    tiled = Variable(torch.FloatTensor([tiled, tiled, tiled, tiled]))

    hidden = rnn.initHidden(4)
    cell = rnn.initCell(4)

    q = rnn(img, tiled, hidden, cell)
    print(q[0])
    print(q[1])

    qq = q[1].data.numpy()
    print(np.sum(qq[2,4,:,:]))

    qqq = q[0].data.numpy()
    print(np.sum(qqq[2,:,2,3]))

    loss_fn = nn.MSELoss()

    print(q[0][0][0][0][0])
    loss = q[0][0][0][0][0]
    print(loss)
    loss.backward()

    optim = torch.optim.Adam(rnn.parameters(), lr = 0.001)
    optim.step()
    print(rnn.num_params()) # Concerning: should be 12.6M...?  Maybe the CDNA is special?
