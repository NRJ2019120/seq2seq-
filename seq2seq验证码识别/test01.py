import tensorflow as tf
import numpy as np
import numpy.random as random
lebal = np.array(random.rand(0,10))


def __one_hot(x):
    # 生成一个4行10列的0矩阵
    z = np.zeros(shape=(4, 10))
    for i in range(4):
        # 取出第i行的值x的下标index，转成int格式
        index = int(x[i])
        # 遍历矩阵，取出每一行的下标
        z[i][index] += 1
    return z
if __name__ == '__main__':
    list_1=[1,2,3,4]
    list_2= [8,6,7,4]
    list_3 = [0,3,6,9]
    z1= __one_hot(list_1)
    z2= __one_hot(list_2)
    z3= __one_hot(list_3)
    list =[]
    list.append(z1)
    list.append(z2)
    list.append(z3)
    lebal= np.stack(list)
    print(lebal.shape)
    list_4=[1,2,3,4]
    list_5= [0,6,7,4]
    list_6 = [0,1,6,9]
    z4= __one_hot(list_4)
    z5= __one_hot(list_5)
    z6= __one_hot(list_6)
    list2 =[]
    list2.append(z4)
    list2.append(z5)
    list2.append(z6)
    lebal2= np.stack(list2)
    print(lebal2.shape)
    sort1 = np.argmax(lebal,axis=2)
    sort2 = np.argmax(lebal2,axis=2)
    print(sort1)
    print(sort2)
    bool = np.equal(sort1,sort2)
    print(bool)
    # print(lebal)
    # acc = np.reduce_mean(tf.cast(bool)