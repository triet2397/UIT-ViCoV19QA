"""Evaluator"""
import torch
from utils.constants import RNN_NAME

class Evaluator(object):
    """Evaluator class"""
    def __init__(self, criterion):
        self.criterion = criterion

    def evaluate(self, model, iterator, teacher_ratio=1.0):
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for _, batch in enumerate(iterator):
                src, src_len = batch.src
                trg = batch.trg
                input_trg = trg if model.name == RNN_NAME else trg[:, :-1]
                output = model(src, src_len, input_trg, teacher_ratio)
                trg = trg.t() if model.name == RNN_NAME else trg[:, 1:]
                output = output.contiguous().view(-1, output.shape[-1])
                trg = trg.contiguous().view(-1)
                # output: (batch_size * trg_len) x output_dim
                # trg: (batch_size * trg_len)
                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
        return epoch_loss / len(iterator)
