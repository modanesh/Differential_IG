import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class TernarizeTanhF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output.data = input.data
        output.round_()
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class TernaryTanh(nn.Module):
    """
    reference: https://r2rt.com/beyond-binary-ternary-and-one-hot-neurons.html
    """

    def __init__(self):
        super(TernaryTanh, self).__init__()

    def forward(self, input):
        output = 1.5 * F.tanh(input) + 0.5 * F.tanh(-3 * input)
        output = ternarizeTanh(output)
        return output


ternarizeTanh = TernarizeTanhF.apply
