import numpy as np
np.random.seed(2)
def foo():
    X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
    y = np.array([[0,1,1,0]]).T
    syn0 = np.random.random((3,3))
    syn1 = np.random.random((3,1))
    #np.random.seed(2)
    for j in xrange(10000):
        l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
        l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
        l2_delta = (y - l2)*(l2*(1-l2))
        l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
        syn1 += l1.T.dot(l2_delta)
        syn0 += X.T.dot(l1_delta)
    return l2

for _ in range(2):
    print foo()

print "Hurrayyyy"
