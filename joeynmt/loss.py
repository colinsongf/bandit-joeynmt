import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class CrossEntropy(_Loss):
    """ Cross-entropy loss, with optional weights on token or sequence level """

    def __init__(self, ignore_index):
        """
        Cross-entropy / neg. log-likelihood loss, summed over the sequence
        :param pad_index:
        :return:
        """
        super(CrossEntropy, self).__init__(reduction='none')
        self.ignore_index = ignore_index

    def forward(self, input, target, weights=None):
        # no reduction
        token_loss = F.nll_loss(input, target, ignore_index=self.ignore_index,
                                reduction='none')
        if weights is not None:
            # multiply by the weights before nll loss
            token_loss *= weights
        return torch.sum(token_loss)

