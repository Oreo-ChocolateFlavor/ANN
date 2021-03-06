import numpy as np
from Activation import ActivationFunction


class Layer:
    def __init__(self,number_members,Input_shape=None,Activation = None,Dropout = 0,Weight_param='Default'):
        self.Input_shape = Input_shape
        self.n_mem = number_members
        self.delayed = False
        self.bias = 0 #np.random.randn(self.n_mem).reshape(1,-1);

        self.Weight_param = Weight_param.lower();

        if(Dropout > 1): raise Exception('Probability exceed 1')
        self.Dropout = Dropout

        if self.Input_shape is not None and self.n_mem is not None:
            self.init_weight();
        else:
            self.delayed = True


        if Activation is None:
            Activation = (lambda x:x,lambda x:1)

        self.Activation = Activation[0];
        self.dActivation = Activation[1];

    def forwardprop(self,prev):
        if prev.shape[1] != self.Input_shape:
            raise Exception('dimension Exception')

        self.prev = prev
        self.bout = self.prev.dot(self.W) + self.bias
        self.aout = self.Activation(self.bout)
        self.aout  = self.aout * np.random.choice(2,self.n_mem,p=[self.Dropout,1-self.Dropout])


        return self.aout

    def backprop(self):
        pass


    def init_weight(self):
        if self.Weight_param == 'xaiver': # recommend for sigmoid & tanh init
            self.W = np.random.randn(self.Input_shape, self.n_mem) / np.sqrt(self.Input_shape)
        elif self.Weight_param =='he': # recommend for relu
            self.bias = 0
            self.W = np.random.randn(self.Input_shape, self.n_mem) / np.sqrt(self.Input_shape/2)
        else:
            self.W = 0.01* np.random.randn(self.Input_shape, self.n_mem)

    def fill_delayed_value(self):
        if self.Input_shape is not None:
            self.init_weight();

if __name__ == '__main__':

    I1 = np.random.randn(10).reshape(-1,10)
    L1 = Layer(Input_shape=I1.shape[1],number_members= 3,Activation=ActivationFunction.Relu)

    print(I1)
    print(L1.forwardprop(I1))
    print(L1.W)
    print(I1.dot(L1.W))
    print(L1.bout)
    print(L1.aout)
