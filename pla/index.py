import random
import time
# from matplotlib.pyplot import plot

# Sign function: returns 1 if x > 0, -1 if x < 0, 0 otherwise
def sgn(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

# Target function: defines the boundary for classification
def f(x, y):
    return sgn(5 * x + 4 * y - 20)

# Training data points on a grid
grid = ([1, 1], [1, 2], [3, 4], [2, 3], [4, 1], [5, 2], [6, 3], [7, 4], [8, 5], [9, 6])

# Initial weights and bias for the perceptron
wt = [0, 0]
bias = 20

# Perceptron output function: computes the sign of the weighted sum plus bias
def g(x, y):
    return sgn(wt[0] * x + wt[1] * y + bias)

# Perceptron function: computes the sign of the weighted sum minus bias
def perceptron(pos):
    ans = pos[0] * wt[0] + pos[1] * wt[1]
    return sgn(ans - bias)

# Learning rate for weight updates
LR = 0.001

# Training function for the perceptron algorithm
def training():
    i = 0  # Counter for iterations
    while True:
        misclassified = []  # List to hold misclassified points
        for data in grid:
            index = 0
            # Check if the data point is misclassified
            if f(data[0], data[1]) != g(data[0], data[1]):
                misclassified.append(data)
                index += 1
        # If no misclassified points, training is complete
        if not misclassified:
            print("Done!")
            break
        print(len(misclassified), " misclassified points")
        # Randomly choose a misclassified point to update weights
        x1, y1 = random.choice(misclassified)
        # Update weights based on the chosen point and learning rate
        wt[0] += f(x1, y1) * x1 * LR
        wt[1] += f(x1, y1) * y1 * LR
        i += 1
        print("W0 : ", wt[0], "  W1 : ", wt[1], "  iterations : ", i)
        print("")

# Run the training function
training()
