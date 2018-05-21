import numpy as np

class Activtion:

    @staticmethod
    def Relu(x):
        return  x*(x>0)

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

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

if __name__ == '__main__':

    print(np.sum(Activtion.softmax(np.array([-1,-2,3,4,5]).reshape(1,-1))))
