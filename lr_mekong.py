import numpy as np
import pandas as pd
from pandas import DataFrame,Series,read_csv
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def pretty_print_linear(coefs, names = None, sort = False):
    if names is None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)

def load_data():
    dataset = read_csv('DataMekongTanChau.csv', usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14], engine='python')
    place = dataset.columns[:-1]
    values = dataset.values
    values = values.astype('float32')
    # training set size
    train_size = 372
    test_size = len(dataset) - train_size
    train = values[0:train_size, :]
    test = values[train_size:len(dataset), :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    return train_X,train_y,place,test_X,test_y

# def scale_data(X):
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#     return X

# def split_data(X,Y):
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
#     return X_train, X_test, Y_train, Y_test

def root_mean_square_error(y_pred,y_test):
    # rmse_train = np.sqrt(np.dot(abs(y_pred-y_test),abs(y_pred-y_test))/len(y_test))
    rmse_train = np.sqrt(mean_squared_error(y_pred, y_test))
    return rmse_train

def r2(y_pred,y_test):
    r2 = r2_score(y_pred, y_test)
    return r2

def plot_real_vs_predicted(y_pred,y_test):
    # plt.plot(y_pred,y_test,'ro')
    # plt.plot([0,5],[0,5], 'g-')
    # plt.xlabel('predict')
    # plt.ylabel('original')
    plt.plot(y_pred, 'r', label='predict')
    plt.plot(y_test, 'g', label='original')
    plt.legend(loc='upper left')
    plt.show()
    return plt

# def generate_regression_values(model, X, y):
#     params = np.append(model.intercept_, model.coef_)
#     predictions = model.predict(X)
#     newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
#     MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

#     # Note if you don't want to use a DataFrame replace the two lines above with
#     # newX = np.append(np.ones((len(X),1)), X, axis=1)
#     # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

#     var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
#     sd_b = np.sqrt(var_b)
#     ts_b = params / sd_b

#     p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

#     sd_b = np.round(sd_b, 3)
#     ts_b = np.round(ts_b, 3)
#     p_values = np.round(p_values, 3)
#     params = np.round(params, 4)

#     myDF3 = pd.DataFrame()
#     myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3[
#         "Probabilites"
#     ] = [params, sd_b, ts_b, p_values]
#     print(myDF3)

def main():
    X_train,Y_train,place,X_test,Y_test = load_data()
    np.set_printoptions(precision=2, linewidth=100, suppress=True, edgeitems=2)
    # X_train = scale_data(X_train)
    # X_test = scale_data(X_test)
    # Create linear regression object
    linreg = LinearRegression()

    # Train the model using the training sets
    linreg.fit(X_train,Y_train)

    print ("Linear model: ", pretty_print_linear(linreg.coef_, place, sort = True))

    # Predict the values using the model
    Y_lin_predict = linreg.predict(X_test)

    # Print the root mean square error 
    print (f"Root Mean Square Error: {root_mean_square_error(Y_lin_predict,Y_test)}")
    # Print R2 score
    print(f"R^2: {r2(Y_lin_predict,Y_test)}")
    # generate_regression_values(linreg, X_test, Y_test)
    plot_real_vs_predicted(Y_lin_predict,Y_test)

if __name__ == '__main__':
    main()
