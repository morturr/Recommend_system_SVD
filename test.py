
import csv
from math import sqrt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


def err(y, pred, mean, p, q, bias_u, bias_m):
    for u in range(1, n_users):
        for m in range(1, n_movies):
            err = 0
            prev_rmse = float('inf')
            _rmse = float('inf')
            diff = 1
            for iter in range(50):
                if diff < 0.1:
                    break
                prev_rmse = _rmse
                pred[u, m] = mean + bias_u[u, :] + bias_m[m, :] + np.dot(np.transpose(q[:, m]), p[u, :])
                err = y[u, m] - pred[u, m]
                _rmse = rmse(y, pred)
                bias_u[u, :] += _gamma * (err - _lambda * bias_u[u, :])
                bias_m[m, :] += _gamma * (err - _lambda * bias_m[m, :])
                q[:, m] += _gamma * (err * p[u, :] - _lambda * q[:, m])
                p[u, :] += _gamma * (err * q[:, m] - _lambda * p[u, :])
                diff = abs(prev_rmse - _rmse)



def rmse(y, pred):
    sub = [(a_i - b_i) ** 2 for a_i, b_i in zip(csr_matrix.toarray(y), pred)]
    sum_sqr = sum(sum(sub))
    size = y.shape[0]*y.shape[1]
    return sqrt(sum_sqr / size)


def my_rmse(y, pred):
    sub = [abs(a_i - b_i) for a_i, b_i in zip(y, pred)]
    sum_sqr = sum(sub)
    return sum_sqr / len(y)



## defs

df = pd.read_csv('D:\\ex1\\ml-20m\\ratingsTrain.csv', nrows=10000)
print 'done reading'
train, validation = train_test_split(df, test_size=0.2)


mp = df['userId']
users = df['userId']
n_users = len(users)
movies = df['movieId']
n_movies = len(movies)
ratings = df['rating']

mat = csr_matrix((ratings, (users, movies)))


constant = 2
p = np.empty((n_users, constant))
p[:] = 0.01
q = np.empty((constant, n_movies))
q[:] = 0.01

_lambda = 0.02
_gamma = 0.005

bias_u = np.empty((n_users, 1))
bias_u[:] = 0.01
bias_m = np.empty((n_movies, 1))
bias_m[:] = 0.01

pred = np.empty(shape=mat.shape)

mean = float(sum(ratings))/len(ratings)
err(mat, pred, mean, p, q, bias_u, bias_m)


pass
