import numpy as np

class Optimizers:

    def __init__(self,optimizer):
        self.optimizer = self.setOptimizer(optimizer)
        self.epchos = 0;

        self.v_ada  = None# vector list for ada , rmsprop , ada
        self.v_moment = None # vector list for momentum ,
        self.isinit = False # is init momentum vectors
        self.epsilon = 1e-8

    def Adam(self,lr,Layers,dLayers,hyperParam=(0.9,0.999)):
        self.epchos += 1;

        if not self.isinit:
            self.v_ada = []
            self.v_moment = []
            for layer in reversed(Layers):
                self.v_ada.append(np.zeros((layer.Input_shape,layer.n_mem)))
                self.v_moment.append(np.zeros((layer.Input_shape, layer.n_mem)))
            self.isinit = True

        for num,(dlayer,cur_layer) in enumerate(zip(dLayers,reversed(Layers))):
            self.v_moment[num] = hyperParam[0] * self.v_moment[num] + (1-hyperParam[0])*dlayer
            self.v_ada[num] = hyperParam[1] * self.v_ada[num] + (1-hyperParam[1]) * (dlayer * dlayer)

            adjusted_v_moment = self.v_moment[num] / (1-hyperParam[0]**self.epchos)
            adjusted_v_ada = self.v_ada[num] / (1-hyperParam[1]**self.epchos)

            cur_layer.W -= lr * adjusted_v_moment / (np.sqrt(adjusted_v_ada) + self.epsilon)


    def RMSProp(self,lr,Layers,dLayers):
        self.epchos += 1;


    def Adagrad(self,lr,Layers,dLayers):
        self.epchos += 1;


    def Default(self,lr,Layers,dLayers):
        self.epchos += 1;
        for dlayer,cur_layer in zip(dLayers,reversed(Layers)):
            cur_layer.W -= lr * dlayer;


    def setOptimizer(self,optimizer):

        optdict = {'default': self.Default, 'adam': self.Adam, 'rmsprop': self.RMSProp,
                   'adagrad': self.Adagrad}

        if not isinstance(optimizer,str):
            raise Exception('optimizier')

        if optimizer not in optdict.keys():
            raise Exception('{} is not supported optimazier'.format(optimizer))

        return optdict[optimizer]
