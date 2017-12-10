import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

lr_rate = 0.001

KERNEL_SIZE = 5

def num_params(model):
    ans = 0
    for param in model.parameters():
        sz = param.size()
        here = 1
        for dim in range(len(sz)):
            here *= sz[dim]
        ans += here
    return ans

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
        

class CNN(nn.Module):
    # as described in Appendix B, except that Appendix B doesn't specify anything about
    # how the results of the memory query are incorporated into the architecture.
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        
        input_channels = in_channels
        self.conv1a = nn.Conv2d(input_channels, 8, 1, padding = 0)
        self.conv1b = nn.Conv2d(input_channels, 8, 3, padding = 1)
        self.conv1c = nn.Conv2d(input_channels, 8, 5, padding = 2)
        self.conv1d = nn.Conv2d(input_channels, 8, 7, padding = 3)

        # CAREFUL HERE
        # the text specifies that the batchnorm happens before the concatenation...
        self.batchnorm1A = nn.BatchNorm2d(8)
        self.batchnorm1B = nn.BatchNorm2d(8)
        self.batchnorm1C = nn.BatchNorm2d(8)
        self.batchnorm1D = nn.BatchNorm2d(8)

        self.batchnorm2A = nn.BatchNorm2d(8)
        self.batchnorm2B = nn.BatchNorm2d(8)
        self.batchnorm2C = nn.BatchNorm2d(8)
        self.batchnorm2D = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1) # dimension-halving
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.conv2a = nn.Conv2d(32, 8, 1, padding = 0)
        self.conv2b = nn.Conv2d(32, 8, 3, padding = 1)
        self.conv2c = nn.Conv2d(32, 8, 5, padding = 2)
        self.conv2d = nn.Conv2d(32, 8, 7, padding = 3)

        self.conv4 = nn.Conv2d(32, 64, 3, stride = 2, padding = 1)

        self.dense = nn.Linear(7 * 7 * 64, 64)
        
    def forward(self, x):
        # input x should have been pre-processed already: saturations in [-1, 1]
        # shape of x is batch_size x input_channels x 28 x 28
        layer1 = F.relu(torch.cat((self.batchnorm1A(self.conv1a(x)),
                                   self.batchnorm1B(self.conv1b(x)),
                                   self.batchnorm1C(self.conv1c(x)),
                                   self.batchnorm1D(self.conv1d(x))),
                                  dim = 1)
                        )
        layer2 = F.relu(self.batchnorm2(self.conv2(layer1)))

        layer3 = F.relu(torch.cat((self.batchnorm2A(self.conv2a(layer2)),
                                   self.batchnorm2B(self.conv2b(layer2)),
                                   self.batchnorm2C(self.conv2c(layer2)),
                                   self.batchnorm2D(self.conv2d(layer2))),
                                  dim = 1)
                        )
        layer4 = self.conv4(layer3)
                                   
        return self.dense(layer4.view([-1, 64 * 7 * 7])).view([-1, 2, 32])
    
        
class TransposeCNN(nn.Module):
    # According to the paper, this should be the transpose of the CNN given above.
    def __init__(self, out_ch):
        # input size: BATCH_SIZE (= 10) x 32
        super(TransposeCNN, self).__init__()
        self.dense = nn.Linear(32, 64 * 7 * 7)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 2, stride = 2)
        
        self.batchnorm2A = nn.BatchNorm2d(32)
        self.batchnorm2B = nn.BatchNorm2d(32)
        self.batchnorm2C = nn.BatchNorm2d(32)
        self.batchnorm2D = nn.BatchNorm2d(32)

        self.batchnorm1A = nn.BatchNorm2d(out_ch)
        self.batchnorm1B = nn.BatchNorm2d(out_ch)
        self.batchnorm1C = nn.BatchNorm2d(out_ch)
        self.batchnorm1D = nn.BatchNorm2d(out_ch)

        self.deconv2a = nn.ConvTranspose2d(8, 32, 1, padding = 0)
        self.deconv2b = nn.ConvTranspose2d(8, 32, 3, padding = 1)
        self.deconv2c = nn.ConvTranspose2d(8, 32, 5, padding = 2)
        self.deconv2d = nn.ConvTranspose2d(8, 32, 7, padding = 3)

        
        self.deconv1a = nn.ConvTranspose2d(8, out_ch, 1, padding = 0)
        self.deconv1b = nn.ConvTranspose2d(8, out_ch, 3, padding = 1)
        self.deconv1c = nn.ConvTranspose2d(8, out_ch, 5, padding = 2)
        self.deconv1d = nn.ConvTranspose2d(8, out_ch, 7, padding = 3)

        
        self.deconv2 = nn.ConvTranspose2d(32, 32, 2, stride = 2)

        self.batchnorm2 = nn.BatchNorm2d(32)
        
    def forward(self, z):
        layer4 = self.dense(z).view([-1, 64, 7, 7])
        layer3 = F.relu(self.deconv4(layer4))

        layer3a, layer3b, layer3c, layer3d = torch.split(layer3, split_size = 8, dim = 1)
        layer2 = F.relu(self.batchnorm2A(self.deconv2a(layer3a)) +
                        self.batchnorm2B(self.deconv2b(layer3b)) +
                        self.batchnorm2C(self.deconv2c(layer3c)) +
                        self.batchnorm2D(self.deconv2d(layer3d))
        )
        layer1 = F.relu(self.batchnorm2(self.deconv2(layer2)))
        layer1a, layer1b, layer1c, layer1d = torch.split(layer1, split_size = 8, dim = 1)
        layer0 = (self.batchnorm1A(self.deconv1a(layer1a)) +
                  self.batchnorm1B(self.deconv1b(layer1b)) +
                  self.batchnorm1C(self.deconv1c(layer1c)) +
                  self.batchnorm1D(self.deconv1d(layer1d)))
        
        return layer0

class Prior(nn.Module):
    def __init__(self, read_heads):
        super(Prior, self).__init__()
        self.read_heads = read_heads
        self.cnn = CNN(read_heads)
        self.tcnn = TransposeCNN(1)
        
    def forward(self, memory_output):
        restored_imgs = [torch.squeeze(self.tcnn(memory_output[r])) for r in range(self.read_heads)]
        #print(restored_imgs[0].size())
        ans = torch.unbind(self.cnn(torch.stack(restored_imgs, dim = 1)), dim = 1)
        # MAYBE: add a skip connection later, for faster / better training?
        return ans

if __name__ == '__main__':
    cnn = CNN(6)
    print(num_params(cnn))

    qe = Variable(torch.FloatTensor(np.random.randn(10, # batch size is 10
                                                    6, 28, 28)))
    c = cnn(qe)
    #print(c[0])
    #print(c[1])

    tcnn = TransposeCNN(2)
    print(num_params(tcnn))

    qe = Variable(torch.FloatTensor(np.random.randn(10,
                                                    32)))
    c = tcnn(qe)
    #print(c)

    p = Prior(read_heads = 5)
    print(num_params(p))
    qe = Variable(torch.FloatTensor(np.random.randn(10,
                                                    32)))
    c = p([qe, qe, qe, qe, qe])
    #print(c[0])
    
    
if __name__ == '___main__':

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
