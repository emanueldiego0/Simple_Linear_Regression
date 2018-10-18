# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 09:23:49 2018

@author: EMANUEL
"""
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time

warnings.filterwarnings('ignore')

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        
        totalError += (y - (m * x + b)) ** 2
    
    return totalError / float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    
    plt.ion()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    line, = ax.plot(points[:,0], m * points[:,0] + b, color='r')
    plt.grid()
    
    for i in range(num_iterations):
        
        b, m = step_gradient(b, m, np.array(points), learning_rate)
        err = compute_error_for_line_given_points(b, m, points)
        print('Iteration: {0}\tm = {1}\tb = {2}\tMSE: {3}'.format(i+1, m, b, err))
        #plt.xlim(20,90)
        plt.scatter(points[:,0], points[:,1], color='cyan')
        #plt.plot(points[:,0], m * points[:,0] + b, color='r')
        #plt.annotate("m = {0}".format(m), xy=(80, 110))
        #plt.annotate("b = {0}".format(b), xy=(80, 100))
        #plt.annotate("err = {}".format(err), xy=(80, 90))
        #plt.show()
        #time.sleep(1)
        #plt.close()
        line.set_ydata(m * points[:,0] + b)
        plt.draw()
        plt.pause(1e-17)
        time.sleep(0.1)
    
    print('\n')
    plt.show()
    return [b, m]


def step_gradient(b_current, m_current, points, learning_rate):
    
    b_gradient = 0
    m_gradient = 0
    
    N = float(len(points))
    
    
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


#hereeee
    
def linear_model(m, b, x):
    return m * x + b


def mean_squared_error(y_true, y_pred):
    N = len(y_true)
    
    sum = 0
    for i in range(N):
        sum += (y_true - y_pred) ** 2
        
    return sum / float(N)
 
    
def MSE_diff_m():
    pass


def MSE_diff_b():
    pass

 
def generate_univariate_data(m, b, scalar_eps = 1, size = 100):
    
    points_x = np.random.normal(size=size)
    points_y = linear_model(m, b, points_x)
    #points_y = m * points_x + b
    
    #bring the noise
    poinst_eps = np.random.randn(size) * scalar_eps
    poinst_y = points_y + poinst_eps
    
    points = np.array([[x,y] for x,y in zip(points_x, poinst_y)])
    
    return points


def plot_univariate_data(points):
    
    plt.scatter(points[:,0], points[:,1])
    plt.grid()
    plt.show()


def run():
    
    points = np.genfromtxt('https://raw.githubusercontent.com/llSourcell/linear_regression_live/master/data.csv', delimiter=',')
    
    #generate random data
    original_m = 200
    original_b = -52.5
    eps_factor = 45
    n_samples = 500
    points = generate_univariate_data(original_m, original_b, eps_factor, n_samples)
    
    #hyperparameters
    learning_rate = 0.005
    initial_b = 1000
    initial_m = -200
    num_iterations = 300
    
    print('[+] Starting Gradient Descent\n[+] Hyperparameters: \n\tInitial b = {0}\n\tInitial m = {1}\n\tNº of iterations = {2}\n\tLearning Rate = {3}\n[+] Initial MSE = {4}'.format(
            initial_b, initial_m, num_iterations, learning_rate, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print('\n')
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print('[+] Ending Gradient Descent\n[+] Results: \n\tFinal b = {0}\n\tFinal m = {1}\n\tFinal MSE = {2}\n\tFinal Iteration = {3}'.format(
            b, m, compute_error_for_line_given_points(b, m, points),num_iterations))
    print('\n----\n')
    print('[+] Original parameters values: m = {0} and b = {1}'.format(original_m, original_b))

if __name__ == '__main__':
    run()