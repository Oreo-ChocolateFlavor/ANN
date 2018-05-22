from Layer import Layer
import numpy as np
import Activation
import Loss
from keras.datasets import boston_housing

from keras.models import Sequential
from keras.layers import Dense

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
        self.Loss_function = Loss_function[0];
        self.dLoss_function = Loss_function[1];
        self.optimizer = optimizer

    def predict(self,input_data):
        for i in self.Layers:
            if not i.delayed:
                input_data = i.forwardprop(input_data)
            else:
                raise Exception("Somevalue doesn't filled")
        return input_data

    def train(self,train_X,train_y,epochs = 200, batch_size=None,verbose = False):

        if train_X.shape[1] != self.Layers[0].Input_shape:
            raise Exception("input_data & input_layer dimension is not equal")

        if batch_size is None: # batch-size가 안주어지면 mini-batch를 사용하지않음
            batch_size = train_X.shape[0]

        for iter_num in range(epochs):

            print('epochs' + str(iter_num))
            print('*'*50)

            start_ind = 0
            while(start_ind < train_X.shape[0]):

                end_idx = min(train_X.shape[0]+1,start_ind+batch_size)
                mini_batchX = train_X[start_ind:end_idx]
                mini_batchy =  train_y[start_ind:end_idx]

                predicted_y = self.predict(mini_batchX) # 예측해서
                Loss = self.Loss_function(mini_batchy,predicted_y) #Loss 계산하고
                self.backprop(mini_batchy,predicted_y) # backpropagation 돌린다.
                start_ind += batch_size
                if verbose:
                    print('Loss:',Loss)
                    print('-'*50)



    def backprop(self,train_y,predicted_y):
        minibatch_dout = self.dLoss_function(train_y,predicted_y)
        dLayers = list()

        for d_out_num,cur_d_out in enumerate(minibatch_dout):
            for layer_num,cur_layer in enumerate(reversed(self.Layers)):
                doutb_dwij = np.full(cur_layer.W.shape,1) * cur_layer.prev[d_out_num:d_out_num+1].T
                douta_doutb = cur_layer.dActivation(cur_layer.bout[d_out_num:d_out_num+1])

                dlayer = cur_d_out * douta_doutb* doutb_dwij

                cur_d_out = dlayer.sum(axis=1).T

                if d_out_num==0:
                    dLayers.append(dlayer)
                else:
                    dLayers[layer_num] += dlayer

        for dlayer in dLayers:
            dlayer = dlayer / minibatch_dout.shape[0]

        for dlayer,cur_layer in zip(dLayers,reversed(self.Layers)):
            cur_layer.W = cur_layer.W - self.lr * dlayer


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    #Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(x_train[:1],y_train[:1],epochs=1,batch_size=1,verbose=1)

    print(model.predict(x_train[1:2]),y_train[1:2])

    y_train = y_train.reshape(-1,1)

    model = ANN()
    model.add(Layer(1,Input_shape=13))
    model.compile(Loss_function=Loss.MSE,optimizer=None,learning_rate=0.0000001)
    model.train(x_train[:1],y_train[:1],epochs=1,verbose=True,batch_size=1)
    print(model.predict(x_train[1:2]))


