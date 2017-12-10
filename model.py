import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

lr_rate = 0.001

CONTROLLER_SIZE = 64

BATCH_SIZE = 10

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
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.i2f = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2I = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2O = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2C = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        forget = F.sigmoid(self.i2f(combined))
        I = F.sigmoid(self.i2I(combined))
        O = F.sigmoid(self.i2O(combined))
        C = forget * cell + I * F.tanh(self.i2C(combined))
        H = O * F.tanh(C)
        return H, C

    def initHidden(self):
        return Variable(torch.zeros(BATCH_SIZE, self.hidden_size).cuda(), requires_grad = False)

    def initCell(self):
        return Variable(torch.zeros(BATCH_SIZE, self.hidden_size).cuda(), requires_grad = False)

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
    # Paper says nothing about how the memory is used in the architecture...
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

class Posterior(nn.Module):
    # TODO: this is not tested
    def __init__(self, read_heads):
        super(Posterior, self).__init__()
        self.read_heads = read_heads
        self.cnn = CNN(read_heads + 1)
        self.tcnn = TransposeCNN(1)

    def forward(self, x, memory_output):
        restored_imgs = [torch.squeeze(self.tcnn(memory_output[r])) for r in range(self.read_heads)]
        restored_imgs.append(x)
        ans = self.cnn(torch.stack(restored_imgs, dim = 1))
        return torch.unbind(ans, dim = 1) # separates means and variances
    
class Likelihood(nn.Module):
    # Paper says very little about this, so we do as we see fit
    def __init__(self, read_heads):
        super(Likelihood, self).__init__()
        self.read_heads = read_heads
        self.mem_tcnn = TransposeCNN(2)
        self.z_tcnn = TransposeCNN(2)
        self.combo = nn.Parameter(torch.randn(6).cuda(), requires_grad = True)
        
    def forward(self, z, memory_output):
        restored_imgs = [self.mem_tcnn(memory_output[r]) for r in range(self.read_heads)]
        restored_z = self.z_tcnn(z)
        ans = restored_z * self.combo[self.read_heads]
        #print(self.combo)
        for i in range(self.read_heads):
            ans += self.combo[i] * restored_imgs[i]
        ans = F.tanh(ans)
        #print(ans)
        return torch.unbind(ans, dim = 1) # separates means and variances

class Attention(nn.Module):
    # Not much discussion in paper about this architecture, either.
    def __init__(self, seq_len):
        super(Attention, self).__init__()
        self.dense0 = nn.Linear(CONTROLLER_SIZE, 128)
        self.dense1 = nn.Linear(128, 128)
        self.dense2 = nn.Linear(128, seq_len)

    def forward(self, h):
        ans = self.dense2(F.relu(self.dense1(F.relu(self.dense0(h)))))
        # softplus, as specified in the paper
        ans = F.softplus(ans)
        return ans / torch.unsqueeze(torch.sum(ans, dim = 1), dim = 1)

class Unified(nn.Module):
    def __init__(self, read_heads, seq_len):
        super(Unified, self).__init__()
        self.read_heads = read_heads
        self.seq_len = seq_len
        
        self.prior = Prior(read_heads = read_heads)
        self.posterior = Posterior(read_heads = read_heads)
        self.likelihood = Likelihood(read_heads = read_heads)
        self.attention1 = Attention(seq_len)
        self.attention2 = Attention(seq_len)
        self.attention3 = Attention(seq_len)
        self.attention4 = Attention(seq_len)
        self.attention5 = Attention(seq_len)
        self.g1 = nn.Parameter(torch.randn(1).cuda(), requires_grad = True)
        self.g2 = nn.Parameter(torch.randn(1).cuda(), requires_grad = True)
        self.g3 = nn.Parameter(torch.randn(1).cuda(), requires_grad = True)
        self.g4 = nn.Parameter(torch.randn(1).cuda(), requires_grad = True)
        self.g5 = nn.Parameter(torch.randn(1).cuda(), requires_grad = True)
        
        self.attentions = [(self.attention1, self.g1),
                           (self.attention2, self.g2),
                           (self.attention3, self.g3),
                           (self.attention4, self.g4),
                           (self.attention5, self.g5)]
        #self.attentions = [(Attention(seq_len = seq_len).cuda(), Variable(torch.randn(1).cuda(), requires_grad = True)) for _ in range(read_heads)]
        self.rnn = LSTM(input_size = 32, hidden_size = CONTROLLER_SIZE)

    def kl(self, g1, g2):
        ans = -0.5 + g2[1] - g1[1] + (torch.exp(g1[1] * 2) + (g1[0] - g2[0]) * (g1[0] - g2[0])) * 0.5 * torch.exp(-2 * g2[1])
        return ans
    
    def forward(self, x_seq):
        print([self.g1, self.g2, self.g3, self.g4, self.g5])
        loss = 0.0 # negative variational lower bound
        
        # BATCH_SIZE (= 10) x seq_len x 28 x 28
        x_seq = torch.unbind(x_seq, dim = 1) # remember, this is a BATCH
        
        controller_hidden = self.rnn.initHidden()
        controller_cell = self.rnn.initCell()
        memory = [Variable(torch.zeros(BATCH_SIZE, 32).cuda(), requires_grad = False) for _ in range(self.seq_len)]

        predicted_last_5 = []
        kls = []
        for s in range(self.seq_len):
            # query the memory (Eqn. 10 in paper)
            mem_output = []
            for att, g in self.attentions:
                w = att(controller_hidden)
                phi = torch.squeeze(torch.matmul(torch.unsqueeze(w, dim = 1), torch.stack(memory, dim = 1)))
                mem_output.append(phi * F.sigmoid(g))
                
            z_distr = self.posterior(x_seq[s], mem_output)
            sampled_z = z_distr[0] + Variable(torch.randn(BATCH_SIZE, 32).cuda(), requires_grad = False) * torch.exp(z_distr[1])

            x_distr = self.likelihood(sampled_z, mem_output)
            if s >= self.seq_len - 5:
                predicted_last_5.append(x_distr[0])
                
            loss += torch.sum(x_distr[1] + np.log(2 * np.pi) +
                              0.5 * (x_distr[0] - x_seq[s]) * (x_distr[0] - x_seq[s]) * torch.exp(x_distr[1] * -2))
            
            
            kl_here = torch.sum(self.kl(z_distr, self.prior(mem_output)))
            kls.append(kl_here)
            loss += kl_here
            # update the memory
            memory[s] = sampled_z
            
            # update the controller (Eqn. 9 in paper)
            controller_hidden, controller_cell = self.rnn(sampled_z, controller_hidden, controller_cell)
        print(loss)
        print(sum(kls))
        return loss, predicted_last_5, kls
    

# MISCELLANEOUS TESTING - IGNORE IT
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
    l = Likelihood(read_heads = 5)
    print(num_params(l))
    c = l(qe, [qe, qe, qe, qe, qe])
    #print(c)
    a = Attention(seq_len = 10)
    f = Variable(torch.randn(2, 64))
    #print(a(f))
    b = Attention(seq_len = 10)
    #print(b(f))
    
    u = Unified(read_heads = 5, seq_len = 25)
    print(num_params(u))
    x_seq = Variable(torch.randn(10, 25, 28, 28))
    print(u(x_seq))
    exit(0)
    d = Variable(torch.randn(1), requires_grad = True)
    r = Variable(torch.randn(2), requires_grad = False)
    print(d)
    print(r)
    r[0] = d
    print(r)
    e = F.mse_loss(r[0], Variable(torch.zeros(1)))
    e.backward()
    print(d.grad.data)
