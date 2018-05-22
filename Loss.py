import numpy as np

class LossFunction:
    @staticmethod
    def MAE(real_output,predicted_output):
        return 1/2 * np.sum(np.abs(real_output-predicted_output))

    @staticmethod
    def MSE(real_output,predicted_output):
        return 1/2*np.power((real_output - predicted_output),2).sum()

    @staticmethod
    def binary_cross_entropy():
        pass

    @staticmethod
    def categorical_cross_entropy():
        pass

class Derivative_LossFuntion:
    @staticmethod
    def d_MAE(real_output, predicted_output):
        pass

    @staticmethod
    def d_MSE(real_output, predicted_output):
        return (predicted_output - real_output)

    @staticmethod
    def d_binary_cross_entropy():
        pass

    @staticmethod
    def d_categorical_cross_entropy():
        pass

MSE = (LossFunction.MSE, Derivative_LossFuntion.d_MSE)


if __name__ == '__main__':
    print(LossFunction.MAE(np.array([1,2,3]),np.array([1,2,6])))
    print(LossFunction.MSE(np.array([1,2,3]),np.array([1,2,6])))