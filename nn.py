import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

import random
from Tensor.micrograd import Value

class Module:
    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0

class Neuron(Module):
    def __init__(self, n_input):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_input)]
        self.bias = Value(0.0)

    def __call__(self, x):
        act = self.bias
        for wi, xi in zip(self.weights, x):
            act += wi * xi
        return act

    def parameters(self):
        return self.weights + [self.bias]


class RELU(Module):
    def __call__(self, x):
        if isinstance(x, list):
            return [xi.relu() for xi in x]
        else:
            return x.relu()

class Tanh(Module):
    def __call__(self, x):
        return self.util(x)

    def util(self, x):
        if isinstance(x, list):
            return [xi.tanh() for xi in x]
        else:
            return x.tanh()

class Linear(Module):
    def __init__(self, input_size, output_size, **kwargs):
        self.neurons = [Neuron(input_size, **kwargs) for _ in range(output_size)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

class RMSLoss(Module):
    def __call__(self, pred, target):
        if not isinstance(pred, list):
            pred = [pred]
        if not isinstance(target, list):
            target = [target]
        losses = [(p - t) ** 2 for p, t in zip(pred, target)]
        return sum(losses) / len(losses)

class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
    
    def step(self):
        for p in self.params:
            p.data -= p.grad * self.lr
        