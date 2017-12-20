import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from model import Unified, BATCH_SIZE

import mnist

lr_rate = 0.001

class Trainer:
    def __init__(self, unified):
        self.unified = unified
        
        self.optimizer = torch.optim.Adam(unified.parameters(), lr = lr_rate)
        self.writer = SummaryWriter()
        self.epoch = 0

        self.train_data = mnist.train_images()
        self.num_train_data = np.shape(self.train_data)[0]
        
    def make_batch(self):
        ans = []
        for i in range(10):
            seq = []
            for j in range(20):
                seq.append(self.train_data[np.random.randint(self.num_train_data)])
            for j in range(5):
                seq.append(seq[j])
            ans.append(seq)
        ans = np.array(ans, dtype=np.float32)
        ans = ans / 256.0
        ans = (ans - 0.5) * 2.0 # centered at 0, range [-1, 1]
        ans = Variable(torch.FloatTensor(ans).cuda())
        return ans
        

    def train(self):
        self.epoch += 1
        x_batch = self.make_batch()
        self.optimizer.zero_grad()
        
        loss, reconstructed_imgs, predicted_last_5, kls = self.unified(x_batch)
        loss.backward()
        self.optimizer.step()
        if self.epoch % 30 == 1:
            reconstr = torch.unbind(torch.stack(reconstructed_imgs, dim = 1), dim = 0)
            predictions = torch.unbind(torch.stack(predicted_last_5, dim = 1), dim = 0)
            curr_batch = torch.unbind(x_batch, dim = 0)
            for b in range(BATCH_SIZE):
                # send it to range (0,1)
                seq = vutils.make_grid(torch.unsqueeze(predictions[b].data.cpu(), dim = 1) / 2.0 + 0.5)
                self.writer.add_image('train_batch ' + str(b), seq, self.epoch)
            for b in range(BATCH_SIZE):
                seq = vutils.make_grid(torch.unsqueeze(curr_batch[b].data.cpu(), dim = 1) / 2.0 + 0.5)
                self.writer.add_image('x_batch ' + str(b), seq, self.epoch)
            for b in range(BATCH_SIZE):
                seq = vutils.make_grid(torch.unsqueeze(reconstr[b].data.cpu(), dim = 1) / 2.0 + 0.5)
                self.writer.add_image('reconstr ' + str(b), seq, self.epoch)
                
        self.writer.add_scalar('loss', loss.data.cpu().numpy(), self.epoch)
        self.writer.add_scalar('log_loss', np.log(loss.data.cpu().numpy()), self.epoch)
        print([kl.data.cpu().numpy()[0] for kl in kls])
def main():
    unified = Unified(read_heads = 5, seq_len = 25).cuda()
    trainer = Trainer(unified)
    while True:
        trainer.train()
    
if __name__ == '__main__':
    main()
