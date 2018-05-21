from Layer import Layer
import numpy as np
from Activation import Activtion
class ANN:
    def __init__(self):
        self.Layers = list()
    def add(self,L:Layer):

        if not isinstance(L,Layer):
            raise TypeError

        if len(self.Layers)>0 :
            if L.Input_shape is None:
                L.delayed = False
                L.Input_shape = self.Layers[-1].n_mem
                L.fill_delayed_value()

            if L.Input_shape != self.Layers[-1].n_mem:
                raise Exception("Layer Dimesion doesn't matched")
        else:
            if L.Input_shape is None:
                raise Exception("Layer Dimesion doesn't matched")

        self.Layers.append(L)

    def compile(self,Loss_function,optimizer,learning_rate = 0.01): #
        self.lr = learning_rate
        self.Loss_function = Loss_function
        self.optimizer = optimizer

    def predict(self,input_data):
        for i in self.Layers:
            if not i.delayed:
                input_data = i.forwardprop(input_data)
            else:
                raise Exception("Somevalue doesn't filled")
        return input_data

    def train(self,train_X,train_y,epochs = 200, batch_size=None,verbose = False):

        if batch_size is None:
            batch_size = train_X.shape[0]

        for iter_num in range(epochs):
            prev_ind = 0
            while(prev_ind <= train_X.shape[0]):
                predicted_y = self.predict(train_X[prev_ind:min(train_X.shape[0]+1,prev_ind+batch_size)]) # 예측해서
                Loss = self.Loss_function(train_y,predicted_y)
                self.backprop()
                prev_ind += train_X.shape[0]

                if verbose:
                    print('Loss:',Loss)


    def backprop(self):
        pass
if __name__ == '__main__':
    model = ANN()
    model.add(Layer(number_members=5,Input_shape=10,Activation=Activtion.Relu))
    model.add(Layer(number_members=2,Activation=Activtion.Relu))
    model.add(Layer(10,Activation=Activtion.Relu))
    I1 = np.random.randn(10).reshape(-1,10)
    print(model.predict(I1))