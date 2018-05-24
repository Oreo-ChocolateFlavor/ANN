from Layer import Layer
import numpy as np
import Activation
import Loss
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


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
        self.Loss_function = Loss_function[0]
        self.dLoss_function = Loss_function[1]
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
            if iter_num%100 == 0:
                print('iter:',iter_num)
            if(verbose):
                print('epochs' + str(iter_num))
                print('*'*50)

            start_ind = 0
            while(start_ind < train_X.shape[0]):

                end_idx = min(train_X.shape[0]+1,start_ind+batch_size)
                mini_batchX = train_X[start_ind:end_idx]
                mini_batchy =  train_y[start_ind:end_idx]

                predicted_y = self.predict(mini_batchX) # 예측해서
                Loss = self.Loss_function(mini_batchy,predicted_y) #Loss 계산하고

                if verbose:
                    print('before Loss:',Loss,'before predicted',predicted_y,'real_output',mini_batchy)

                self.backprop(mini_batchy,predicted_y) # backpropagation 돌린다.
                start_ind += batch_size
                if verbose:
                    predicted_y = self.predict(mini_batchX)  # 예측해서
                    Loss = self.Loss_function(mini_batchy, predicted_y)  # Loss 계산하고
                    print('after Loss:',Loss,'after predicted',predicted_y,'real_output',mini_batchy)
                    print('Loss:',Loss)
                    print('-'*50)



    def backprop(self,train_y,predicted_y):
        minibatch_dout = self.dLoss_function(train_y,predicted_y)
        dLayers = list()

        for d_out_num,cur_d_out in enumerate(minibatch_dout):
            for layer_num,cur_layer in enumerate(reversed(self.Layers)):
                doutb_dwij = np.full(cur_layer.W.shape,1) * cur_layer.prev[d_out_num:d_out_num+1].T
                douta_doutb = cur_layer.dActivation(cur_layer.bout[d_out_num:d_out_num+1])

                temp_result = cur_d_out * douta_doutb

                dlayer =  temp_result * doutb_dwij
                cur_d_out = temp_result * cur_layer.W

                cur_d_out = cur_d_out.sum(axis=1)
                dlayer = dlayer / minibatch_dout.shape[0]
                if d_out_num==0:
                    dLayers.append(dlayer)
                else:
                    dLayers[layer_num] += dlayer

        for dlayer,cur_layer in zip(dLayers,reversed(self.Layers)):
            cur_layer.W = cur_layer.W - (self.lr * dlayer)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    model = Sequential()
    model.add(Dense(1, input_dim=13, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(1,kernel_initializer='normal', activation='relu'))

    #Compile model
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.0000001))

    #x_train = np.array([[1, 1], [0, 1], [1, 1], [0, 1], [-1, 1], [0, 1], [-2, 1]])
    #y_train = np.array([3, 1, 2, 1, -1, 1, -3]).reshape(-1, 1)

    history = model.fit(x_train,y_train,epochs=10,batch_size=1,verbose=1)
    #print(model.predict(np.array([[1,1]])))
    #x = np.array([0.05,0.1]).reshape(-1,2)
    #y = np.array([0.01,0.99]).reshape(-1,2)




    #y_train = y_train.reshape(-1,1)
    #model = ANN()
    #model.add(Layer(1,Input_shape=13,Activation=Activation.Relu))
    #model.add(Layer(1,Activation=Activation.Relu))
    #model.add(Layer(6,Activation=Activation.sigmoid))
    #model.add(Layer(1))

    #model.compile(Loss_function=Loss.MSE,optimizer=None,learning_rate=0.0000001)
    ##model.Layers[0].W = np.array([[0.15,0.25],[0.20,0.30]]);
    #model.Layers[0].bias = np.array([0.35,0.35])
    #model.Layers[1].W = np.array([[0.40, .50], [0.45, 0.55]]);
    #model.Layers[1].bias = np.array([0.6, 0.6])
    #model.train(x_train,y_train,epochs=1000,verbose=False,batch_size=1)



    print(y_test - model.predict(x_test).reshape(-1))


