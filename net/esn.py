import numpy as np
import numpy.random as rnd
import torch
import torch, torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms


class GenESN(nn.Module):


    def __init__(self,
                 n_in, n_res, n_out,
                 ro_hidden_layers=1,
                 lin_size=64,
                 density=0.2,
                 spec_radius=0.99,
                 leaking_rate=1.0,
                 in_scaling=1.0,
                 dropout_rate=0.1,
                 batch_size=300,
                 device=torch.device('cuda'),
                 is_feedback=False,
                 fb_scaling=1.0,
                 **params):
        super(GenESN, self).__init__()

        self.n_in         = n_in
        self.n_res        = n_res
        self.n_out        = n_out
        self.device       = device
        self.batch_size   = batch_size
        self.is_feedback  = is_feedback
        self.leaking_rate = leaking_rate

        if self.is_feedback:
            self.prev_input = torch.zeros(batch_size, n_out, requires_grad=False).to(self.device)

        # reservoir initiation
        try:
            self.w_input = params['w_input']
            self.w_res   = params['w_res']
            print('External reservoir set')

        except:

            self.w_input = self.initiate_in_reservoir(n_res, n_in, scaling=in_scaling).to(device)
            self.w_res   = self.initiate_reservoir(density, n_res, spec_radius, device).float()
            self.w_fb    = self.initiate_fb_reservoir(n_res, n_out, scaling=fb_scaling).to(device)

            print('Internal reservoir set')
            n_non_zero = self.w_res[self.w_res > 0.01].shape[0]
            print('Reservoir has {} non zero values ({:.2%})' \
                    .format(n_non_zero, n_non_zero / (n_res ** 2)))

        # readout layers
        self.readout_in       = nn.Linear(n_res, lin_size)
        self.hidden_ro_layers = [nn.Linear(lin_size, lin_size).to(device) for i in range(ro_hidden_layers - 1)]
        self.readout_out      = nn.Linear(lin_size, self.n_out)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.LogSoftmax()

        return


    def forward(self, input, hidden_state):
        state = torch.mm(self.w_input, input.T) + torch.mm(self.w_res, hidden_state)
        hidden_state =  hidden_state * (1 - self.leaking_rate)
        if self.is_feedback:
            hidden_state += self.leaking_rate * torch.tanh(state + torch.mm(self.w_fb, self.prev_input.T))
            self.prev_input = input
        else:
            hidden_state += self.leaking_rate * torch.tanh(state)

        output = self.readout_in(hidden_state.T)
        if self.hidden_ro_layers:
            first_output = output.clone()
            for layer in self.hidden_ro_layers:
                output = self.dropout(layer(output))
            output += first_output
        output = self.readout_out(output)

        return   hidden_state, self.softmax(output)


    def init_hidden(self):
        """hidden state initiation"""
        return Variable(torch.zeros(self.n_res, self.batch_size, requires_grad=False)).to(self.device)


    def initiate_in_reservoir(self, n_reservoir, n_input, scaling):

        w_input = np.random.rand(n_reservoir, n_input) - 0.5
        w_input = w_input * scaling
        w_input = torch.tensor(w_input, dtype=torch.float32, requires_grad=False)

        return w_input


    def initiate_fb_reservoir(self, n_reservoir, n_output, scaling=1.0):

        w_fb = np.random.rand(n_reservoir, n_output) - 0.5
        w_fb *= scaling
        w_fb = torch.tensor(w_fb, dtype=torch.float32, requires_grad=False)
        return w_fb

    def initiate_reservoir(self, density, n_res, spec_radius, device):

      w_res = np.identity(n_res)
      w_res = np.random.permutation(w_res)
      w_res = torch.tensor(w_res, requires_grad=False).to(device)

      number_nonzero_elements = density * n_res * n_res
      while np.count_nonzero(w_res.cpu().data) < number_nonzero_elements:
            q = torch.tensor(self.create_rotation_matrix(n_res), requires_grad=False).to(device)
            w_res = torch.mm(q, w_res)

      w_res *= spec_radius

      return w_res


    def create_rotation_matrix(self, n_reservoir):

        h   = rnd.randint(0, n_reservoir)
        k   = rnd.randint(0, n_reservoir)
        phi = rnd.rand(1) * 2 * np.pi

        Q       = np.identity(n_reservoir)
        Q[h, h] = np.cos(phi)
        Q[k, k] = np.cos(phi)
        Q[h, k] = - np.sin(phi)
        Q[k, h] = np.sin(phi)

        return Q

    def add_hidden_emb(self, sequence):
        self.embedding = torch.cat((self.embedding, sequence.cpu()), 0)


    def finalize(self):
        self.init_embedding = False
