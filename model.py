import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

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
def kl_div(gauss1, gauss2):
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

    def initHidden(self, use_cuda):
	if use_cuda:
            return Variable(torch.zeros(BATCH_SIZE, self.hidden_size).cuda())
	else:
            return Variable(torch.zeros(BATCH_SIZE, self.hidden_size))

    def initCell(self, use_cuda):
	if use_cuda:
            return Variable(torch.zeros(BATCH_SIZE, self.hidden_size).cuda())
	else:
            return Variable(torch.zeros(BATCH_SIZE, self.hidden_size))
	    
class CNN(nn.Module):
    # as described in Appendix B
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        
        input_channels = in_channels
        self.conv1a = nn.Conv2d(input_channels, 8, 1, padding = 0)
        self.conv1b = nn.Conv2d(input_channels, 8, 3, padding = 1)
        self.conv1c = nn.Conv2d(input_channels, 8, 5, padding = 2)
        self.conv1d = nn.Conv2d(input_channels, 8, 7, padding = 3)

        # the text specifies that the batchnorm happens before the concatenation
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
    # Paper says nothing about how the memory is used in the architecture.
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
    def __init__(self, read_heads, use_memory = False):
        super(Posterior, self).__init__()
        self.use_memory = use_memory
        # Turns out the posterior doesn't (for these sequences) actually require any knowledge about the memory.
        # Eliminating the dependency should speed up training.
        self.read_heads = read_heads if use_memory else 0
        self.cnn = CNN(1)
        self.dense1 = nn.Linear((2 + self.read_heads) * 32, 128)
        self.dense2 = nn.Linear(128, 128)
        self.dense3 = nn.Linear(128, 64)

    def forward(self, x, memory_output = None):
        z = torch.squeeze(self.cnn(torch.unsqueeze(x, dim = 1))).view([-1, 2, 32])
        if self.use_memory:
            if memory_output is None:
                print("This Posterior predictor was initialized with use_memory = True; pass in the memory output.")
            memory_output_and_z = torch.stack(memory_output + list(torch.unbind(z, dim = 1)), dim = 1)
        else:
            memory_output_and_z = z
        layer1 = F.relu(self.dense1(memory_output_and_z.view([-1, (self.read_heads + 2) * 32])))
        layer2 = F.relu(self.dense2(layer1))
        ans = self.dense3(layer2).view([-1, 2, 32])
        return torch.unbind(ans, dim = 1) # separates means and variances; returns (mean, log sigma)

class Likelihood(nn.Module):
    def __init__(self):
        super(Likelihood, self).__init__()
        self.z_tcnn = TransposeCNN(2)
    def forward(self, z):
        ans = self.z_tcnn(z)
        ans = F.tanh(ans)
        return torch.unbind(ans, dim = 1) # separates means and variances; returns (mean, log sigma)
    
class Attention(nn.Module):
    # No discussion in paper about this architecture, either.
    def __init__(self, read_heads, seq_len, experiment_type):
        super(Attention, self).__init__()
        self.read_heads = read_heads
        self.seq_len = seq_len
        self.dense0 = nn.Linear(CONTROLLER_SIZE, 128)
        self.dense1 = nn.Linear(128, 128)
        # EXPERIMENT CONTROL:
        # 0 is to just replicate the original paper
        # 1 is to use sparsity regularization
        # 2 is to attempt to avoid sparsity
        self.type = experiment_type
        if self.type == 2:
            self.dense2 = nn.Linear(128, seq_len * read_heads, bias = False)
        else:
            self.dense2 = nn.Linear(128, seq_len * read_heads)
        
    def forward(self, h):
        # returns a pair (attention_weights, regularization loss)
        # [attention_weights] is a batch_size x seq_len x read_heads tensor.
        layer0 = self.dense2(F.relu(self.dense1(F.relu(self.dense0(h)))))
        layer0 = layer0.view([-1, self.seq_len, self.read_heads])
        # See the comment above initialization of [self.type].
        loss = 0.0
        if self.type == 0:
            # softplus, as specified in the paper
            ans = F.softplus(layer0)
            ans = ans / torch.unsqueeze(torch.sum(ans, dim = 1), dim = 1)
        if self.type == 1:
            alpha = 0.001 # This works well, it seems (based on a few experiments).
            ans = F.softplus(layer0)
            ans = ans / torch.unsqueeze(torch.sum(ans, dim = 1), dim = 1)
            loss = alpha * torch.sum(torch.sqrt(ans + 1e-6))
        if self.type == 2:
            # here, we allow coefficients not to sum to 1, but heavily discourage it
            ans = layer0
            beta = 10.0
            gap = torch.sum(ans, dim = 1) - 1
            loss = beta * torch.sum(gap * gap)
            
        return (ans, loss)
        
class MemoryGate(nn.Module):
    def __init__(self, read_heads):
        super(MemoryGate, self).__init__()
        # These are the gating mechanism for the memory, the correction biases (see Eqn. 12 in the paper).
        self.dense0 = nn.Linear(CONTROLLER_SIZE, 128)
        self.dense1 = nn.Linear(128, 128)
        self.dense2 = nn.Linear(128, read_heads)
        
    def forward(self, h):
	# returns a batch_size x read_heads tensor
        ans = self.dense2(F.relu(self.dense1(F.relu(self.dense0(h)))))
        return ans
    
class Unified(nn.Module):
    def __init__(self, read_heads, seq_len, experiment_type, use_cuda):
        super(Unified, self).__init__()
        self.read_heads = read_heads
        self.seq_len = seq_len
	self.use_cuda = use_cuda
        
        self.prior = Prior(read_heads = read_heads)
        self.posterior = Posterior(read_heads = read_heads, use_memory = False)
        self.likelihood = Likelihood()
        self.attention = Attention(read_heads = read_heads, seq_len = seq_len, experiment_type = experiment_type)
        self.memory_gate = MemoryGate(read_heads)
        
        self.rnn = LSTM(input_size = 32, hidden_size = CONTROLLER_SIZE)
    
    def forward(self, x_seq):
        loss = 0.0 # negative variational lower bound, plus (maybe) some regularization
        
        # BATCH_SIZE (= 10) x seq_len x 28 x 28
        x_seq = torch.unbind(x_seq, dim = 1) # remember, this is a BATCH
        
        controller_hidden = self.rnn.initHidden(self.use_cuda)
        controller_cell = self.rnn.initCell(self.use_cuda)
	if self.use_cuda:
            memory = [Variable(torch.zeros(BATCH_SIZE, 32).cuda()) for _ in range(self.seq_len)]
        else:
            memory = [Variable(torch.zeros(BATCH_SIZE, 32)) for _ in range(self.seq_len)]
	     
	reconstructed_imgs = []
        predicted_last_5 = []
        kls = []
        for s in range(self.seq_len):
            # query the memory (Eqn. 10 in paper)
            mem_output = []
            attention_weights, regularize_term = self.attention(controller_hidden)
            gate_weights = self.memory_gate(controller_hidden)
            loss += regularize_term
            
	    phi = torch.bmm(torch.transpose(attention_weights, 1, 2), torch.stack(memory, dim = 1))
	    
	    mem_output = torch.unbind(phi * torch.unsqueeze(gate_weights, dim = 2), dim = 1)
            
	    z_distr = self.posterior(x_seq[s])
            sampled_z = z_distr[0] + Variable(torch.randn(BATCH_SIZE, 32, out = phi.data.new()), requires_grad = False) * torch.exp(z_distr[1])

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

        return loss, reconstructed_imgs, predicted_last_5, kls
    
