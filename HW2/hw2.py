'''
NTHU EE Machine Learning HW2
Author: 
Student ID: 
'''
import numpy as np
import pandas as pd
import math
import scipy.stats
import argparse

def phi(data, O1, O2):
    x1_max = np.max(data[:, 0])
    x1_min = np.min(data[:, 0])
    x2_max = np.max(data[:, 1])
    x2_min = np.min(data[:, 1])
    s1 = (x1_max - x1_min) / (O1 - 1)
    s2 = (x2_max - x2_min) / (O2 - 1)
    count = 0
    for x1, x2, x3 in data[:, :3]:
        row = np.array([])
        for i in range(1, O1 + 1):
            mui = s1 * (i - 1) + x1_min
            for j in range(1, O2 + 1):
                muj = s2 * (j - 1) + x2_min
                temp = math.exp(- math.pow((x1 - mui), 2) / (2 * math.pow(s1, 2))  \
                                - math.pow((x2 - muj), 2) / (2 * math.pow(s2, 2)))
                row = np.append(row, temp)
        row = np.append(row, x3)
        row = np.append(row, 1.0)
        if count == 0:
            phi_data = row
            count += 1
        else:
            phi_data = np.vstack((phi_data, row))                
    
    return phi_data

# do not change the name of this function
def BLR(train_data, test_data_feature, O1=2, O2=2):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    alpha = 0.1
    beta = 9
    phi_data = phi(train_data, O1, O2)
    target = train_data[:, 3]
    SN_inverse = alpha * np.identity(O1*O2 + 2) + beta * np.matmul(np.transpose(phi_data), phi_data)
    mn = beta * np.matmul(np.matmul(np.linalg.inv(SN_inverse), np.transpose(phi_data)), target)
    phi_test = phi(test_data_feature, O1, O2)
    y_BLRprediction = np.matmul(phi_test, mn)
    return y_BLRprediction 


# do not change the name of this function
def MLR(train_data, test_data_feature, O1=2, O2=2):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    phi_data = phi(train_data, O1, O2)
    target = train_data[:, 3]
    weight = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi_data), phi_data)), np.transpose(phi_data)), target)
    phi_test = phi(test_data_feature, O1, O2)
    y_MLLSprediction = np.matmul(phi_test, weight)
    return y_MLLSprediction 


def CalMSE(data, prediction):

    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean__squared_error = sum_squared_error/prediction.shape[0]

    return mean__squared_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O1', '--O_1', type=int, default=6)
    parser.add_argument('-O2', '--O_2', type=int, default=6)
    args = parser.parse_args()
    O_1 = args.O_1
    O_2 = args.O_2
    
    data_train = pd.read_csv('Training_set.csv', header=None).to_numpy()
    data_test = pd.read_csv('Validation_set.csv', header=None).to_numpy()
    data_test_feature = data_test[:, :3]
    data_test_label = data_test[:, 3]
    
    predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)

    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR, data_test_label), e2=CalMSE(predict_MLR, data_test_label)))


if __name__ == '__main__':
    main()