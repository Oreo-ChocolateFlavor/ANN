import numpy as np

class ActivationFunction:
    @staticmethod
    def Relu(x):
        return x*(x>0)
    @staticmethod
    def sigmoid(x):
        converted = np.clip(x,-500,500)
        return 1/(1+np.exp(-converted))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def softmax(x):
        expoential = np.exp(x)
        return expoential/np.sum(expoential,axis=1).reshape(-1,1)

    @staticmethod
    def step(x):
        ret_x = x
        ret_x[ret_x <= 0] = 0
        ret_x[ret_x > 0] = 1

        return ret_x

class Derivative_ActivationFunction:
    @staticmethod
    def d_sigmoid(x,*args,**kwargs):

        ret = ActivationFunction.sigmoid(x)

        return ret*(1-ret)

    @staticmethod
    def d_Relu(x,*args,**kwargs):
        return x >0

        pass
    @staticmethod
    def d_tanh(x,*args,**kwargs):
        return 1-np.power(ActivationFunction.tanh(x),2)

    @staticmethod
    def d_softmax(x,prev_out_d,*args,**kwargs):
        expoential = np.exp(x)
        S = np.sum(expoential)
        d_S = np.sum(prev_out_d*expoential*(-1/np.square(S)))
        return (1/S *expoential,d_S*expoential);

    @staticmethod
    def step(x):
        pass

Relu = (ActivationFunction.Relu,Derivative_ActivationFunction.d_Relu)
sigmoid = (ActivationFunction.sigmoid,Derivative_ActivationFunction.d_sigmoid)
tanh = (ActivationFunction.tanh,Derivative_ActivationFunction.d_tanh)
softmax = (ActivationFunction.softmax,Derivative_ActivationFunction.d_softmax)
using_final_result = True

if __name__ == '__main__':

    print(np.sum(ActivationFunction.softmax(np.array([-1,-2,3,4,5]).reshape(1,-1))))
