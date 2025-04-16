
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make the data

x = [0.11957891, 0.55069399, 0.70009694, 0.57481472, 0.23121113, 0.4647737,
 0.98952893, 0.96795901,]

y = [0.92906457, 0.47133868, 0.54697896, 0.327777,   0.81730965, 0.70370025,
 0.70845996, 0.0117534,]

# x coord, y coord, type
data = [[x[0], y[0], 1],
        [x[1], y[1], 1],
        [x[2], y[2], 0],
        [x[3], y[3], 0],
        [x[4], y[4], 1],
        [x[5], y[5], 1],
        [x[6], y[6], 0],
        [x[7], y[7], 0],] 

unkown = [0.568933, 0.678554]


w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

learning_rate = 0.3
track_costs = []

for i in range(70000):
    r = np.random.randint(len(data))
    point = data[r]

    method = point[0] * w1 + point[1] * w2  + b
    result = sigmoid(method)

    d_result = float(d_sigmoid(method))
    target = point[2]


    #cost function >>> (result - target) ** 2

    cost = (result - target) ** 2
    d_cost = 2 * (result - target)
    
    track_costs.append(cost)

    #my partial deravatives from method

    d_w1 = point[0] 
    d_w2  = point[1]
    d_b = 1

    #chaining

    d_cost_w1 = d_cost * d_result * d_w1
    d_cost_w2 = d_cost * d_result * d_w2
    d_cost_b = d_cost * d_result * d_b

    #updating parameters

    w1 = w1 - learning_rate * d_cost_w1
    w2 = w2 - learning_rate * d_cost_w2
    b = b - learning_rate * d_cost_b

for i in range(len(data)):
    point = data[i]
    print(point)
    
    method = point[0] * w1 + point[1] * w2  + b
    result = sigmoid(method)

    print(result)
    
    
    method_1 = unkown[0] * w1 + unkown[1] * w2  + b
    result_1 = sigmoid(method_1)

    print(result_1 )




#scatter data for visual

for i in range (len(data)):
    
    point = data[i]
    color = 'r'

    if point[2] == 0:
        color = 'b' 

    plt.scatter(point[0], point[1], c = color)



plt.scatter(unkown[0], unkown[1], c = 'g')
plt.scatter( 0.70845996, 1 , c = 'o')
plt.show()