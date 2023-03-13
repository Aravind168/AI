import random
import numpy as np

cities = []
n = 0
with open('input.txt', 'r') as f:
    n = int(f.readline().strip())
    for _ in range(int(n)):
        cities.append([int(x)
                       for x in f.readline().strip().split()])

cities = np.array(cities)
size = 100
population = np.empty(shape=[size, n])
matingPool = np.empty(shape=[size//2, n])
elite = 15
costs = np.zeros(size)
scores = np.zeros(size)


def createInitialPopulation():
    for i in range(size):
        population[i] = np.array(np.random.permutation(n))


def calculateDistance(city1, city2):
    city1 = int(city1)
    city2 = int(city2)
    return ((cities[city1][0])-cities[city2][0]**2 + (cities[city1][1])-cities[city2][1]**2 + (cities[city1][2])-cities[city2][2]**2)**0.5


def calculateCost(path):
    dist = 0
    for i in range(len(path)-1):
        print(path[i])
        dist += calculateDistance(path[i], path[i+1])
    else:
        dist += calculateDistance(path[i], path[0])
    return dist


def calculateCosts():  # entire population
    global scores
    dist = 0
    for i in range(len(population)-1):
        dist += calculateDistance(population[i][0], population[i+1][0])
    else:
        dist += calculateDistance(population[i][0], population[0][0])
    scores[i] = dist


def rankPaths():
    global scores,costs
    costs=scores
    scores = 1/scores
    summ = np.sum(scores)
    scores /= summ
    scores = np.cumsum(scores)
    print(scores)


def selection():
    rnd = random.random()
    global scores
    return np.where(scores > rnd)[0][0]


def createMatingPool():  # requires scores[i]
    for i in range(size//2):
        matingPool[i] = selection()


def best(parent1, parent2):
    dist1 = calculateCost(parent1)
    dist2 = calculateCost(parent2)
    return min(dist1, dist2)


def crossover(parent1, parent2):  # two-point crossover -> returns child or best parent
    if random.random() > 0.3:
        start = random.randint(0, len(parent1)-2)
        end = random.randint(start+1, len(parent1)-1)
        child = np.zeros(len(parent1))
        for i in range(start, end+1):
            child[i] = parent1[i]
        idx = end - start
        for i in range(len(parent2)):
            if parent2[i] not in child:
                child[idx] = parent2[i]
                idx += 1
        return child
    else:
        return best(parent1, parent2)


def mutate():
    for i in range(elite+1, len(population)):
        if random.random() < 0.05:
            idx1 = random.randint(1, n-1)
            idx2 = random.randint(1, n-1)
            population[i][idx1], population[i][idx2] = population[i][idx2], population[i][idx1]


def sortArrays():
    global population, costs, scores
    # population, costs, scores = (np.array(t) for t in zip(*sorted(zip(population, costs,scores), key = costs)))
    common = costs.argsort()
    costs.sort()
    population = population[common]
    scores = scores[common]


def evolve():
    newpop = []
    global population
    for i in range(elite):
        newpop.append(population[i])
    for i in range(len(matingPool)-1):
        newpop.append(crossover(matingPool[i], matingPool[i+1]))
        newpop.append(crossover(matingPool[i], matingPool[i+1]))
    newpop.append(crossover(population[0], population[-1]))
    newpop.append(crossover(population[-1], population[0]))
    population = np.array(newpop)


def printOutput(path):
    with open('output.txt', 'w') as f:
        for i in range(len(path)):
            f.write(" ".join(map(str, cities[path[i]]))+"\n")
        f.write(" ".join(map(str, cities[path[0]]))+"\n")


createInitialPopulation()
i = 0
while i < 1200:
    calculateCosts()
    rankPaths()
    sortArrays()
    createMatingPool()
    evolve()
    mutate()
    i += 1
printOutput(population[0])
