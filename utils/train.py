import numpy as np
from random import sample
import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from IPython.display import clear_output


def trainESN(net,
             data,
             tokens,
             token_to_id,
             max_length,
             n_epoch,
             batch_size,
             opt,
             lr_scheduler,
             history=[],
             device=torch.device('cuda'),
             random_batching=True,
             figsize=(13, 6.5),
             plot_loss=True,
             legend=True):
    """GenESN training loop function"""

    try:
        for i in range(n_epoch):

            if random_batching:
                batch_ix      = to_matrix(sample(data, batch_size), token_to_id, max_len=max_length)
                batch_ix      = torch.tensor(batch_ix, dtype=torch.int64)
                one_hot_batch = torch.nn.functional.one_hot(batch_ix, len(tokens))
            else:
                batch_ix      = to_matrix(data[i * batch_size:(i+1) * batch_size], token_to_id, max_len=max_length)
                batch_ix      = torch.tensor(batch_ix, dtype=torch.int64)
                one_hot_batch = torch.nn.functional.one_hot(batch_ix, len(tokens))

            sequence           = compute_state(net, one_hot_batch)
            predicted_seq      = sequence[:, :-1]
            actual_next_tokens = batch_ix[:, 1:].long().to(device)

            # loss = lm_cross_entropy(predictions_logp,actual_next_tokens)
            loss = - torch.mean(torch.gather(predicted_seq, dim=2, index=actual_next_tokens[:, :, None]))
            loss.backward()
            opt.step()
            opt.zero_grad()

            history.append(loss.data.cpu().numpy())
            lr_scheduler.step(loss)
            if plot_loss:
                if (i + 1) % 100 == 0 or i == 1:
                    plt.figure(figsize=figsize)
                    plt.grid()
                    clear_output(True)
                    if legend:
                        plt.plot(history, '.-',
                            label='loss, lr = {}, epochs={} '.format(opt.param_groups[0]['lr'], i))
                    else:
                        plt.plot(history, '.-', label='loss')
                    plt.legend()
                    plt.title('The loss function of ESN')
                    plt.show()
    except KeyboardInterrupt:
        print('KeyboardInterrupt, stoping...')
        return


def to_matrix(data,
              token_to_id,
              max_len=None,
              dtype='float32',
              batch_first=True):
    """Casts a list of tokens into rnn-digestable matrix"""

    max_len = max_len or max(map(len, data))
    data_ix = np.zeros([len(data), max_len], dtype) + token_to_id[' ']

    for i in range(len(data)):
        line_ix = [token_to_id[c] for c in data[i]]
        data_ix[i, :len(line_ix)] = line_ix

    if not batch_first:
        data_ix = np.transpose(data_ix)

    return data_ix


def compute_state(net, batch_index):
    """GenESN state computing"""

    batch_size, max_length, _ = batch_index.size()

    hid_state = net.init_hidden()
    logprobs  = []

    for x_t in batch_index.transpose(0, 1):
        hid_state, logp_next = net(x_t.float().to(net.device), hid_state)
        logprobs.append(logp_next)

    return torch.stack(logprobs, dim=1)


def lm_cross_entropy(pred, target):

    pred_flat   = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[-1])
    target_flat = target.view(-1)

    return F.cross_entropy(pred_flat, target_flat, ignore_index=0)


def scheduler(optimizer, patience):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      patience=patience,
                                                      factor=0.5,
                                                      verbose=True)
