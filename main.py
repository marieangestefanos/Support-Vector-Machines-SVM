import random

import matplotlib.pyplot as plt
import numpy as np
import random as rd
import math

from scipy.optimize import minimize

#Global variables

# Kernel function
def linear_ker(x, y):
    return np.dot(np.transpose(x), y)

def poly_ker(x, y, p=3):
    return pow(linear_ker(x, y) + 1, p)

def rbf_ker(x, y, sigma=4):
    return np.exp(-np.dot((x-y),(x-y))/(2*sigma*sigma))


def kernel(x, y, meth):
    if meth == "lin":
        return linear_ker(x, y)
    if meth == "poly":
        return poly_ker(x, y)
    if meth == "rad":
        return rbf_ker(x, y)

def objective(alpha):
    return 0.5*np.dot(np.dot(alpha, p), alpha) - np.sum(alpha)

def zerofun(alpha):
    return np.dot(alpha, targets)

def nonzerovalues(alpha, threshold=0.1):
    suppvectors = []
    for i, a in enumerate(alpha):
        if a>threshold:
            suppvectors.append((a, inputs[i], targets[i]))
    return suppvectors

def bias(alpha, meth):
    suppvectors = nonzerovalues(alpha)
    if len(suppvectors) == 0:
        print("There is no support vectors.")
        return None
    i = 0
    while i < len(suppvectors):
        a, s, t = suppvectors[0]
        if a < C:
            return np.sum([alpha[i]*targets[i]*kernel(s, inputs[i], meth) for i in range(N)]) - t
        else:
            i+=1
    print("No value of alpha lower than C")
    return None

def indicator(alpha, suppvector, meth):
    b = bias(alpha, meth)
    if b == None:
        print("Bias not found.")
        return None
    else:
        return np.sum([alpha[i]*targets[i]*kernel(suppvector, inputs[i], meth) for i in range(N)]) - b





## Testing kernel functions
# x = np.array([1, 2, 5, 4])
# y = np.array([5, 2.5, 1, 1.25])
# print(linear_ker(x, y))
# print(poly_ker(x, y, 3))
# print(rbf_ker(x, y, 5))

np.random.seed(100)

classA = np.concatenate(
     (np.random.randn(20, 2)*0.2 + [1.5, 0.5],
     np.random.randn(20, 2)*0.2 + [-1.5, 0.5]))
#     np.random.randn(10, 2)*0.7 + [0.5, -3]))

#classA = np.random.randn(20, 2)*1 + [-1.5, 0.0]
classB = np.random.randn(20, 2)*0.2 + [0.0, -0.5]

inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

N = inputs.shape[0] # Nb of rows/samples

permute = list(range(N))
rd.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]
# meth = "lin"
meth = "poly"
# meth = "rad"

def compute_p(meth):
    p = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            p[i, j] = targets[i]*targets[j]*kernel(inputs[i, :], inputs[j, :], meth)
    return p

start = np.zeros(N)
C = 2
bnds=[(0, C) for b in range(N)] #or C instead of None
cons={'type':'eq', 'fun':zerofun}
p = compute_p(meth)
ret = minimize(objective, start, bounds=bnds, constraints=cons)
alpha = ret.x
flag = ret.success
print(flag)

plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b.')
plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.')
plt.axis('equal')

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)

grid = np.array([[indicator(alpha, [x, y], meth=meth)
                   for x in xgrid]
                   for y in ygrid])

plt.contour(xgrid, ygrid, grid,
            (-1.0, 0.0, 1.0),
            colors = ('red', 'black', 'blue'),
            linewidths=(1, 3, 1))

plt.savefig('img.png')
plt.show()

