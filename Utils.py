import time
import sys
class Utils:
    @staticmethod
    def progress_bar(now_times,max_times,Loss):
        n = int(now_times/max_times*20)
        print('*'*n + '-'*(20-n)+' Loss: {}'.format(Loss))

if __name__ == '__main__':
    print('test')
    for i in range(11):
        Utils.progress_bar(i,10,100)