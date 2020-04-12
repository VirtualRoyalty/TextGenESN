import numpy as np
from random import sample
import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.model_selection import ParameterGrid
from net import esn
from utils import train


def GridSearchOptimizer(net_class,
                        data,
                        max_length,
                        token_to_id,
                        tokens,
                        in_out,
                        learning_rate,
                        n_epoch=1000,
                        batch_size=256,
                        device=torch.device('cuda'),
                        **params):
    try:
        prev_res = 0
        for param in ParameterGrid(params):

            print('Params: {}'.format(param))
            net = net_class( n_in=in_out[0],
                             n_res=param['n_res'],
                             n_out=in_out[1],
                             ro_hidden_layers=param['ro_hidden_layers'],
                             lin_size=param['lin_size'],
                             density=param['density'],
                             leaking_rate=param['leaking_rate'],
                             device=device)
            net.to(device)

            learning_rate = learning_rate
            opt           = torch.optim.Adam(net.parameters(), lr=learning_rate)
            lr_scheduler  = train.scheduler(opt, patience=150)

            history = []
            for i in range(n_epoch):

                batch_ix      = train.to_matrix(sample(data, batch_size), token_to_id, max_len=max_length)
                batch_ix      = torch.tensor(batch_ix, dtype=torch.int64)
                one_hot_batch = torch.nn.functional.one_hot(batch_ix, len(tokens))

                logp_seq           = train.compute_state(net, one_hot_batch)
                predictions_logp   = logp_seq[:, :-1]
                actual_next_tokens = batch_ix[:, 1:].long().to(device)

                loss = -torch.mean(torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:,:,None]))
                # loss = lm_cross_entropy(predictions_logp,actual_next_tokens)

                loss.backward()
                opt.step()
                opt.zero_grad()
                history.append(loss.data.cpu().numpy())
            print('Loss function value: {} \n'.format(round(np.mean(history[-50:]), 3)))
    except KeyboardInterrupt:
        print('KeyboardInterrupt, stoping...')
        return
