import numpy as np
from random import sample
import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable


def generate_sample(net,
                    max_length,
                    tokens,
                    token_to_id,
                    seed_phrase=' ',
                    temperature=1.0,
                    enforce_end=False,
                    device=torch.device('cuda')):

    x_sequence = [token_to_id[token] for token in seed_phrase]
    x_sequence = torch.tensor([x_sequence], dtype=torch.int64).to(device)

    with torch.no_grad():
        hid_state = net.init_hidden()
        for i in range(len(seed_phrase) - 1):
            x_t = x_sequence[:, i]
            x_t = torch.tensor(x_t.data, dtype=torch.int64)
            x_t = torch.nn.functional.one_hot(x_t, len(tokens))
            hid_state, _ = net(x_t.float().to(net.device), hid_state)

        for _ in range(max_length - len(seed_phrase)):
            x_t = x_sequence[:, -1]
            x_t = torch.tensor(x_t.data, dtype=torch.int64)
            x_t = torch.nn.functional.one_hot(x_t, len(tokens))

            hid_state, logp_next = net(x_t.float().to(net.device), hid_state)

            p_next = F.softmax(logp_next.cpu() / temperature, dim=-1).data.numpy()[0]

            next_ix = np.random.choice(len(tokens), p=p_next)
            next_ix = torch.tensor([[next_ix]], dtype=torch.int64)
            if enforce_end and tokens[next_ix] == ' ':
                break
            x_sequence = torch.cat([x_sequence.float(), next_ix.to(device).float()], dim=1)

    return ''.join([tokens[ix] for ix in x_sequence.int().cpu().data.numpy()[0]])
