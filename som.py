import csv
import numpy as np

# Function to find the euclidean distance between the weights and the input data


def findEuclideanDist(num_clusters, W, x):
    result = []
    for i in range(num_clusters):
        tmp = W[:, i]
        diff = np.subtract(tmp, x)
        square = np.square(diff)
        result.append(np.sum(square))
    return result


# Function to find the neighbours of a minimum value
def findNeighbours(index, W, R, clusters):
    min_wt = W[:, index]
    indices = [index]
    for i in range(clusters):
        if(not (i == index)):
            tmp = W[:, i]
            diff = np.subtract(tmp, min_wt)
            square = np.square(diff)
            if(np.sum(square) < R):
                indices.append(i)
    return indices


inputData = []
# Input the data from file
with open("input.csv", "rt") as file:
    data = csv.reader(file)
    for row in data:
        inputData.append(row[0].split(' '))

# Converting each attribute to float value
for i in range(len(inputData)):
    for j in range(len(inputData[i])):
        inputData[i][j] = float(inputData[i][j])

print("The input patterns are:")
ctr = 1
for x in inputData:
    print("P-" + str(ctr), "->", x)
    ctr += 1

num_clusters = int(input("Enter number of output clusters:"))

# contains the random numbers as weights between the units
weights = np.random.rand(len(inputData[0]), num_clusters)/2

max_iterations = 25  # will run for maximum 100 iterations

# generate a list of alpha values equally spaced for 100 iterations
alpha = np.linspace(0.6, 0.01, max_iterations)

R = 1  # update the weights of neighbourhood in the radius of 1 around the minimum example

clustersAssigned = [-1]*len(inputData)
# learning begins
for iteration in range(max_iterations):
    if(iteration == 18):
        # print("Iteration Number " + str(iteration) + ":\n", weights)
        R -= 1
    for x in inputData:
        calculatedDist = findEuclideanDist(num_clusters, weights, x)
        min_index = calculatedDist.index(min(calculatedDist))
        clustersAssigned[inputData.index(x)] = min_index

        # finding the neighbours of the minimum value
        neighbours = findNeighbours(min_index, weights, R, num_clusters)

        # updating the weights of the neighbours
        # according to wij(new) = wij(old) + alpha(xi - wij)
        for i in neighbours:
            tmp = weights[:, i]
            tmp2 = np.multiply(alpha[iteration], np.subtract(x, tmp))
            weights[:, i] = np.add(tmp, tmp2)

print("Final Weights:\n", weights)
print("\nThe Clusters Assigned are:")
for i in range(len(clustersAssigned)):
    print("P" + str(i+1) + " assigned Cluster Number:",  clustersAssigned[i])
