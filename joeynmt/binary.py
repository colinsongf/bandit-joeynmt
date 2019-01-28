import torch
from torch import nn
from torch.autograd import Function

# https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
#https://github.com/Wizaron/binary-stochastic-neurons

class Hardsigmoid(nn.Module):

    def __init__(self):
        super(Hardsigmoid, self).__init__()
        self.act = nn.Hardtanh()

    def forward(self, x):
        return (self.act(x) + 1.0) / 2.0


class RoundFunctionST(Function):
    """Rounds a tensor whose values are in [0, 1] to a tensor with values in {0, 1}"""

    @staticmethod
    def forward(ctx, input):
        """Forward pass
        Parameters
        ==========
        :param input: input tensor
        Returns
        =======
        :return: a tensor which is round(input)"""

        # We can cache arbitrary Tensors for use in the backward pass using the
        # save_for_backward method.
        # ctx.save_for_backward(input)

        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """In the backward pass we receive a tensor containing the gradient of the
        loss with respect to the output, and we need to compute the gradient of the
        loss with respect to the input.
        Parameters
        ==========
        :param grad_output: tensor that stores the gradients of the loss wrt. output
        Returns
        =======
        :return: tensor that stores the gradients of the loss wrt. input"""

        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        # input, weight, bias = ctx.saved_variables
        return grad_output

RoundST = RoundFunctionST.apply

"""
class BinaryNet(Net):

    def __init__(self, mode='Deterministic', estimator='ST'):
        super(BinaryNet, self).__init__()

        assert mode in ['Deterministic', 'Stochastic']
        assert estimator in ['ST', 'REINFORCE']
        #if mode == 'Deterministic':
        #    assert estimator == 'ST'

        self.mode = mode
        self.estimator = estimator

        self.fc1 = nn.Linear(784, 100)
        if self.mode == 'Deterministic':
            self.act = DeterministicBinaryActivation(estimator=estimator)
        elif self.mode == 'Stochastic':
            self.act = StochasticBinaryActivation(estimator=estimator)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, input):
        x, slope = input
        x = x.view(-1, 784)
        x_fc1 = self.act((self.fc1(x), slope))
        x_fc2 = self.fc2(x_fc1)
        x_out = F.log_softmax(x_fc2, dim=1)
        return x_out
"""

class DeterministicBinaryActivation(nn.Module):

    def __init__(self):
        super(DeterministicBinaryActivation, self).__init__()
        self.act = Hardsigmoid()
        self.binarizer = RoundST

    def forward(self, input):
        x, slope = input
        x = self.act(slope * x)
        x = self.binarizer(x)
        return x


class BinarySigmoidLayer(nn.Module):
    # TODO add more options: stochastic, estimator
    def __init__(self):
        super(BinarySigmoidLayer, self).__init__()
        self.act = DeterministicBinaryActivation()

    def forward(self, input):
        # assumption: input contains tensor and slope
        return self.act(input)