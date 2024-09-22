import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

import random
from Tensor.micrograd import Value
class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
class Neuron(Module):

    def __init__(self, nin, activation='Linear'):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.activation = activation
        
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)

        if self.activation == 'ReLU':
            return act.relu()
        elif self.activation == 'Tanh':
            return act.tanh()
        elif self.activation == 'Linear':
            return act
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.activation} Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, activation='Linear'):
        self.neurons = [Neuron(nin,activation) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self):
        self.layers = []

    def add(self,input, output, activation='Linear'):
        self.layers.append(Layer(input, output, activation=activation))
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

class Loss(Module):
    def __call__(self, model, X, y):
        
        inputs = [list(map(Value, xrow)) for xrow in X]
    
        scores = list(map(model, inputs))
        y = [Value(yi) for yi in y]
        losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(y, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))
        
        alpha = Value(1e-4)
        reg_loss = alpha * sum((p*p for p in model.parameters()))
        total_loss = data_loss + reg_loss
        
        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(y, scores)]
        return total_loss, sum(accuracy) / len(accuracy)

class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
    
    def step(self,k):
        self.lr = 1.0 - 0.9*k/100
        for p in self.params:
            p.data -= p.grad * self.lr
    
class RELU(Module):
    
    def __init__(self, input_size, output_size, **kwargs):
        self.neurons = [Neuron(input_size, **kwargs) for _ in range(output_size)]
        
    def __call__(self, x):
        if isinstance(x, list):
            return [xi.relu() for xi in x]
        else:
            return x.relu()
        
    def parameters(self):
        params = []
        
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        
        return params
    
class Tanh(Module):
    
    def __init__(self, input_size, output_size, **kwargs):
        self.neurons = [Neuron(input_size, **kwargs) for _ in range(output_size)]
        
    def __call__(self, x):
        if isinstance(x, list):
            return [xi.tanh() for xi in x]
        else:
            return x.tanh()

    def parameters(self):
        params = []
        
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        
        return params
            
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
    
    def backward(self):
        for neuron in self.neurons:
            for p in neuron.parameters():
                p.backward()
        