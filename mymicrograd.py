import random
class scalar:
    def __init__(self,data,_children=(),_op='',label=''):
        self.data=data
        self.grad=0.0
        self._op=_op
        self._prev=set(_children)
        self.label=label
        self._backward=lambda:None
    def __repr__(self):
        return f"scalar(data={self.data},grad={self.grad})"
    def __add__(self,other):
        other= (other) if isinstance(other,scalar) else scalar(other)
        out=scalar(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad+=out.grad*1.0
            other.grad+=out.grad*1.0
        out._backward=_backward
        return out
    def __neg__(self):
        out=scalar(-self.data,(self,),'neg')
        def _backward():
            self.grad+=-1.0*out.grad
        out._backward=_backward
        return out
    def __sub__(self,other):
        return self+(-other)
    def __mul__(self,other):
        other=(other) if isinstance(other,scalar) else scalar(other)
        out=scalar(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad+=out.grad*other.data
            other.grad+=out.grad*self.data
        out._backward=_backward
        return out
    def __truediv__(self,other):
        out=(self)*((other)**(-1))
        return out
    def __radd__(self,other):
        return self+other
    def __rsub__(self,other):
        return other + (-self)
    def __rmul__(self,other):
        return self*other
    def __rtruediv__(self,other):
        return (self)*((other)**-1)
    def __pow__(self,other):
        out=scalar(self.data**other,(self,),f"self**{other}")
        def _backward():
            self.grad+=(out.grad)*((other)*((self.data)**(other-1)))
        out._backward=_backward
        return out
    def relu(self):
        out=scalar(0.0 if self.data<0 else self.data,(self,),'ReLU')
        def _backward():
            self.grad+=(out.grad * (self.data>0))
        out._backward=_backward
        return out

    def backward(self):
        topo=[]
        visited=set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad=1.0

        for node in reversed(topo):
            node._backward()

class Neuron:
    def __init__(self,nin,nonlin=True):
        self.w=[scalar(random.uniform(-1,1)) for _ in range(nin)]
        self.b=scalar(0)
        self.nonlin=nonlin
    def __call__(self,x):
        out= sum(((wi*xi) for wi,xi in zip(self.w,x)),self.b)
        return out.relu() if self.nonlin else out
    def parameters(self):
        return self.w+[self.b]
class Layer:
    def __init__(self,nin,non,nonlin=True):
        self.neurons=[Neuron(nin,nonlin) for _ in range(non)]
    def __call__(self,x):
        outs=[neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs)==1 else outs
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
class MLP:
    def __init__(self,nin,non):
        z=[nin]+non
        self.layers=[Layer(z[i],z[i+1],nonlin=(i!=len(non)-1)) for i in range(len(z)-1) ]
    def __call__(self,x):
        out=x
        for layer in self.layers:
          out=layer(out)
        return out
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
