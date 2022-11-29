from neupy import algorithms
from cec17_functions import cec17_test_func
import numpy as np
from scipy.optimize import minimize


DIMENSION_NUM = 10
Fun_num = 3
Current_fitness_evaluations = 0


# evaluate the fitness of the incoming individual
def Fitness_Evaluation(individual):
    global Fun_num, Current_fitness_evaluations
    f = [0]
    cec17_test_func(individual, f, DIMENSION_NUM, 1, Fun_num)
    Current_fitness_evaluations = Current_fitness_evaluations + 1
    return f[0]


print(Fitness_Evaluation([ -5.26676453 , -5.37579316 ,-74.29769864 , 38.1246246 ,  29.26436457,
  23.82962633 , 22.0612114 ,  10.97769994 , 27.97330222 ,-99.99830537]))