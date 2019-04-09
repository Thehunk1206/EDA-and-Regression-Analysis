import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def total_error(b,w_matrix,Y_train,X_train):
    totalError = 0
    for j in range(len(Y_train)):
        X = X_train[j]
        y = Y_train[j]
        totalError += (y - ((np.dot(w_matrix,X)) + b)) ** 2
    return totalError / float(len(Y_train))

def gradient_descent_step(current_b,current_w_matrix,X_train,Y_train,learning_rate):
    gradient_b = 0
    gradient_w_matrix=[]
    for i in range(len(X_train[1])):
    	gradient_w_matrix.append(0)
    gradient_w_matrix=np.array(gradient_w_matrix,dtype=float)

    N = float(len(Y_train))

    for j in range(len(Y_train)):
        X = X_train[j]
        y = Y_train[j]
        gradient_b += -(2/N) * (y - ((np.dot(current_w_matrix,X)) + current_b))
        gradient_w_matrix += -(2/N) * X * (y - ((np.dot(current_w_matrix,X)) + current_b))

        new_b = current_b - (learning_rate * gradient_b)
        new_w_matrix = current_w_matrix - (learning_rate * gradient_w_matrix)
    return [new_b, new_w_matrix]


def gradient_descent_engine(X_train,Y_train,starting_b,starting_w_matrix,numOfIteration,learning_rate):
    b = starting_b
    weight_matrix = starting_w_matrix

    for i in range(numOfIteration):
    	b,weight_matrix = gradient_descent_step(b,weight_matrix,X_train,Y_train,learning_rate)
    return [b,weight_matrix]


def main():
    df = pd.read_csv("kc_house_data.csv")
    train_data,test_data = train_test_split(df,train_size=0.5,random_state=3)


    features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
    X_train = np.array(train_data[features],dtype=float)
    Y_train = np.array(train_data['price'],dtype=float)

    X_test = np.array(test_data[features])
    Y_test = np.array(test_data['price'])

    #Y intercept
    initial_b=0
    #creating weight matrix with elements = no. of features
    initial_w_matrix =[]
    for i in range(len(X_train[0])):
        initial_w_matrix.append(0)

    initial_w_matrix=np.array(initial_w_matrix,dtype=float)

    numOfIteration = 1000
    Learning_rate = 0.01

    
    [b,weight_matrix] = gradient_descent_engine(X_train,Y_train,initial_b,initial_w_matrix,numOfIteration,Learning_rate)

    print("After {0} iterations b = {1}, weight_matrix = {2}, error = {3}".format(numOfIteration, b, weight_matrix, total_error(b, weight_matrix, Y_train,X_train)))



if __name__ == '__main__':
    main()