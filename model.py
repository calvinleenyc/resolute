import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

lr_rate = 0.001

CONTROLLER_SIZE = 128

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

# KL Divergence between two gaussians; returns KL(gauss1 || gauss2)
def kl_div(self, gauss1, gauss2):
    # Each gaussian is a pair of vectors (mean, log sigma).
    ans = -0.5 + gauss2[1] - gauss1[1] + \
        (torch.exp(gauss1[1] * 2) + (gauss1[0] - gauss2[0]) * (gauss1[0] - gauss2[0])) * 0.5 * torch.exp(-2 * gauss2[1])
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
        layer4 = F.relu(self.conv4(layer3))
        
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

        self.batchnorm1A = nn.BatchNorm2d(8)
        self.batchnorm1B = nn.BatchNorm2d(8)
        self.batchnorm1C = nn.BatchNorm2d(8)
        self.batchnorm1D = nn.BatchNorm2d(8)

        self.deconv2a = nn.ConvTranspose2d(8, 32, 1, padding = 0)
        self.deconv2b = nn.ConvTranspose2d(8, 32, 3, padding = 1)
        self.deconv2c = nn.ConvTranspose2d(8, 32, 5, padding = 2)
        self.deconv2d = nn.ConvTranspose2d(8, 32, 7, padding = 3)

        
        self.deconv1a = nn.ConvTranspose2d(8, 8, 1, padding = 0)
        self.deconv1b = nn.ConvTranspose2d(8, 8, 3, padding = 1)
        self.deconv1c = nn.ConvTranspose2d(8, 8, 5, padding = 2)
        self.deconv1d = nn.ConvTranspose2d(8, 8, 7, padding = 3)

        
        self.deconv2 = nn.ConvTranspose2d(32, 32, 2, stride = 2)

        self.batchnorm2 = nn.BatchNorm2d(32)

        self.last_conv = nn.Conv2d(32, out_ch, 3, padding = 1)
        
    def forward(self, z):
        layer4 = self.dense(z).view([-1, 64, 7, 7])
        layer3 = F.relu(self.deconv4(layer4))

        layer3a, layer3b, layer3c, layer3d = torch.split(layer3, split_size = 8, dim = 1)
        layer2 = F.relu(self.batchnorm2A(self.deconv2a(layer3a)) +
                        self.batchnorm2B(self.deconv2b(layer3b)) +
                        self.batchnorm2C(self.deconv2c(layer3c)) +
                        self.batchnorm2D(self.deconv2d(layer3d))
        ) + layer3
        layer1 = F.relu(self.batchnorm2(self.deconv2(layer2)))
        layer1a, layer1b, layer1c, layer1d = torch.split(layer1, split_size = 8, dim = 1)
        layer0 = torch.cat((self.batchnorm1A(self.deconv1a(layer1a)),
                            self.batchnorm1B(self.deconv1b(layer1b)),
                            self.batchnorm1C(self.deconv1c(layer1c)),
                            self.batchnorm1D(self.deconv1d(layer1d))), dim = 1)
        
        return self.last_conv(F.relu(layer0))

class Prior(nn.Module):
    # Paper says nothing about how the memory is used in the architecture...
    def __init__(self, read_heads):
        super(Prior, self).__init__()
        self.read_heads = read_heads
        self.dense1 = nn.Linear(read_heads * 32, 128)
        self.dense2 = nn.Linear(128, 128)
        self.dense3 = nn.Linear(128, 64)
        
    def forward(self, memory_output):
        memory_output = torch.stack(memory_output, dim = 1)
        layer1 = F.relu(self.dense1(memory_output.view([-1, self.read_heads * 32])))
        layer2 = F.relu(self.dense2(layer1))
        ans =  self.dense3(layer2).view([-1, 2, 32])
        return torch.unbind(ans, dim = 1)
        
class Posterior(nn.Module):
    def __init__(self, read_heads):
        super(Posterior, self).__init__()
        self.read_heads = read_heads
        self.cnn = CNN(1)
        self.dense1 = nn.Linear((2 + read_heads) * 32, 128)
        self.dense2 = nn.Linear(128, 128)
        self.dense3 = nn.Linear(128, 64)

    def forward(self, x, memory_output):
        tx = torch.squeeze(self.cnn(torch.unsqueeze(x, dim = 1))).view([-1, 2, 32])
        memory_output = torch.stack(memory_output + list(torch.unbind(tx, dim = 1)), dim = 1)
        layer1 = F.relu(self.dense1(memory_output.view([-1, (self.read_heads + 2) * 32])))
        layer2 = F.relu(self.dense2(layer1))
        ans = self.dense3(layer2).view([-1, 2, 32])
        return torch.unbind(ans, dim = 1) # separates means and variances; returns (mean, log sigma)

class Likelihood(nn.Module):
    # Paper says very little about this, so we do as we see fit
    def __init__(self):
        super(Likelihood, self).__init__()
        self.z_tcnn = TransposeCNN(2)
    def forward(self, z):
        ans = self.z_tcnn(z)
        ans = F.tanh(ans)
        return torch.unbind(ans, dim = 1) # separates means and variances; returns (mean, log sigma)
    
class Attention(nn.Module):
    # No discussion in paper about this architecture, either.
    def __init__(self, seq_len):
        super(Attention, self).__init__()
        self.dense0 = nn.Linear(CONTROLLER_SIZE, 128)
        self.dense1 = nn.Linear(128, 128)
        # EXPERIMENT CONTROL:
        # 0 is to just replicate the original
        # 1 is to use sparsity regularization
        # 2 is to attempt to avoid sparsity
        # To adjust the experiment type, we change the line below.
        # (In real code, this could be controlled through a command-line argument to train.py.)
        self.type = 0
        if self.type == 2:
            self.dense2 = nn.Linear(128, seq_len, bias = False)
        else:
            self.dense2 = nn.Linear(128, seq_len)

        
    def forward(self, h):
        ans = self.dense2(F.relu(self.dense1(F.relu(self.dense0(h)))))
        # See the comment above initialization of [self.type].
        if self.type == 0:
            # softplus, as specified in the paper
            ans = F.softplus(ans)
            return (ans / torch.unsqueeze(torch.sum(ans, dim = 1), dim = 1), 0.0)
        if self.type == 1:
            alpha = 0.001
            ans = F.softplus(ans)
            
            ans = ans / torch.unsqueeze(torch.sum(ans, dim = 1), dim = 1)
            return (ans, alpha * torch.sum(torch.sqrt(ans + 1e-6)))
        if self.type == 2:
            # here, we allow coefficients not to sum to 1, but heavily discourage it
            beta = 10.0
            gap = torch.sum(ans, dim = 1) - 1
            loss = beta * torch.sum(gap * gap)
            return (ans, loss)
        
class MemoryGate(nn.Module):
    # This is the gating mechanism for the memory, the correction biases (see eqn. 12 in the paper).
    def __init__(self):
        super(G, self).__init__()
        self.dense0 = nn.Linear(CONTROLLER_SIZE, 128)
        self.dense1 = nn.Linear(128, 128)
        self.dense2 = nn.Linear(128, 1)
    def forward(self, h):
        ans = self.dense2(F.relu(self.dense1(F.relu(self.dense0(h)))))
        return ans
    
class Unified(nn.Module):
    def __init__(self, read_heads, seq_len):
        super(Unified, self).__init__()
        self.read_heads = read_heads
        self.seq_len = seq_len
        
        self.prior = Prior(read_heads = read_heads)
        self.posterior = Posterior(read_heads = read_heads)
        self.likelihood = Likelihood()
        # As specified in the paper, there are exactly 5 read heads.
        # (Better would be to merge the Attention and MemoryGate modules, and have it take in [read_heads]
        # as a parameter, and return mem_output.)
        assert(self.read_heads == 5)
        self.attention1 = Attention(seq_len)
        self.attention2 = Attention(seq_len)
        self.attention3 = Attention(seq_len)
        self.attention4 = Attention(seq_len)
        self.attention5 = Attention(seq_len)
        self.gate1 = MemoryGate()
        self.gate2 = MemoryGate()
        self.gate3 = MemoryGate()
        self.gate4 = MemoryGate()
        self.gate5 = MemoryGate()
        
        self.attentions = [(self.attention1, self.gate1),
                           (self.attention2, self.gate2),
                           (self.attention3, self.gate3),
                           (self.attention4, self.gate4),
                           (self.attention5, self.gate5)]
        
        self.rnn = LSTM(input_size = 32, hidden_size = CONTROLLER_SIZE)
    
    def forward(self, x_seq):
        loss = 0.0 # negative variational lower bound
        
        # BATCH_SIZE (= 10) x seq_len x 28 x 28
        x_seq = torch.unbind(x_seq, dim = 1) # remember, this is a BATCH
        
        controller_hidden = self.rnn.initHidden()
        controller_cell = self.rnn.initCell()
        memory = [Variable(torch.zeros(BATCH_SIZE, 32).cuda(), requires_grad = False) for _ in range(self.seq_len)]
        reconstructed_imgs = []
        predicted_last_5 = []
        kls = []
        for s in range(self.seq_len):
            # query the memory (Eqn. 10 in paper)
            mem_output = []
            for r in range(self.read_heads):
                att, g = self.attentions[r]
                w, regularize_term = att(controller_hidden)
                loss += regularize_term
                
                # Useful debug output to understand how well the training is going.  Eventually, the model learns
                # enough that one of the w's will be a one-hot encoding representing the 3rd position.
                # (Should be controlled by a [verbose] flag.)
                if s == 23:
                    print(w[0])
                    print(g(controller_hidden)[0])
                    
                phi = torch.squeeze(torch.bmm(torch.unsqueeze(w, dim = 1), torch.stack(memory, dim = 1)))
                mem_output.append(phi * F.sigmoid(g(controller_hidden)))
                
            # Turns out the posterior didn't actually require any knowledge about the memory, and so eliminating this
            # dependency should speed up training.
            # Far better would be to just create an appropriately-sized call to torch.zeros(), or have a [use_memory]
            # flag in Posterior.
            z_distr = self.posterior(x_seq[s], [0.0 * x for x in mem_output])
            
            sampled_z = z_distr[0] + Variable(torch.randn(BATCH_SIZE, 32).cuda(), requires_grad = False) * torch.exp(z_distr[1])

            x_distr = self.likelihood(sampled_z)

            reconstructed_imgs.append(x_distr[0])
            
            # negative log gaussian
            loss += torch.sum(x_distr[1] + 0.5 * np.log(2 * np.pi) +
                              0.5 * (x_distr[0] - x_seq[s]) * (x_distr[0] - x_seq[s]) * torch.exp(x_distr[1] * -2))
            
            z_prior = self.prior(mem_output)
            
            kl_here = torch.sum(kl_div(z_distr, z_prior))
            kls.append(kl_here)
            loss += kl_here
            # update the memory
            memory[s] = sampled_z
            
            if s >= self.seq_len - 5:
                # For purposes of viewing training progress, we don't bother sampling from z, just use the mean.
                predicted_last_5.append(self.likelihood(z_prior[0])[0])
            
            # update the controller (Eqn. 9 in paper)
            controller_hidden, controller_cell = self.rnn(sampled_z, controller_hidden, controller_cell)

        # The loss is a sum of two terms (when replicating original), and it's useful to watch which one
        # is larger as training progresses.  (Should be moved to train.py and plotted instead of printed.)
        print(loss)
        print(sum(kls))
        return loss, reconstructed_imgs, predicted_last_5, kls
    
