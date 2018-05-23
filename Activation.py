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
        return np.exp(x)/np.sum(np.exp(x),axis=1)

    @staticmethod
    def step(x):
        ret_x = x
        ret_x[ret_x <= 0] = 0
        ret_x[ret_x > 0] = 1

        return ret_x

class Derivative_ActivationFunction:
    @staticmethod
    def d_sigmoid(x):
        return ActivationFunction.sigmoid(x)*(1-ActivationFunction.sigmoid(x))

    @staticmethod
    def d_Relu(x):
        pass
    @staticmethod
    def d_tanh(x):
        return 1-np.power(ActivationFunction.tanh(x),2)

    @staticmethod
    def d_softmax(x):
        pass

    @staticmethod
    def step(x):
        pass

Relu = (ActivationFunction.Relu,Derivative_ActivationFunction.d_Relu)
sigmoid = (ActivationFunction.sigmoid,Derivative_ActivationFunction.d_sigmoid)

if __name__ == '__main__':

    print(np.sum(ActivationFunction.softmax(np.array([-1,-2,3,4,5]).reshape(1,-1))))
