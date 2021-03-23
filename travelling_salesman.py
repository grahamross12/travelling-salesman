import numpy as np
import random
import matplotlib.pyplot as plt
import copy

# Number of points
N = 30

# Number of iterations
iterations = 1000

# Population size
n_pop = 50
n_children = 30
n_winners = 5

mutation_prob = 0.1

# Plot size
plt.rcParams["figure.figsize"] = (10, 10)

#Set a seed
#random.seed(sys.time)

def main():
    """The main cycle of the algorithm."""

    points = create_points()
    #points = create_circular_points()
    distance_matrix = create_map(points)
    population = create_population()

    for i in range(iterations):
        new_population = []

        # Score the population
        distances = grade_all_solutions(distance_matrix, population)
        n_best = np.argmin(distances)
        best = population[n_best]
        lowest_distance = distances[n_best]
        child = crossover(population[0], population[1])

        # Plot the best candidate
        plot_candidate(points, best, lowest_distance, i+1)

        # Breed new solutions
        for j in range(n_children):
            child = crossover(population[pick_parent(distances)], population[pick_parent(distances)])
            new_population += [child]

        # Mutate new children
        for j in range(len(new_population)):
            new_population[j] = copy.copy(mutate(new_population[j]))

        # Keep members of the pevious generation
        new_population += [population[n_best]]
        for j in range(1, n_winners):
            keeper = pick_parent(distances)
            new_population += [population[keeper]]

        # Add new random memebers
        while len(new_population) < n_pop:
            new_population += [create_candidate()]

        # Replace the old population with a real copy
        population = copy.deepcopy(new_population)

    # After the algorith completes, keep the graph shown
    plt.ioff()
    plt.show()


def create_circular_points():
    """Create N equally spaced points around a circle."""
    r = 40
    points = []
    for theta in np.linspace(0, 2 * np.pi, N):
        x = 50 + r * np.cos(theta)
        y = 50 + r * np.sin(theta)
        point = (x, y)
        points.append(point)
    return points

def create_points():
    """Create N randomly generated points in a 100 x 100 grid."""
    points = []
    for i in range(N):
        point = (random.randint(0, 100), random.randint(0, 100))
        points.append(point)
    return points


def create_map(points):
    """Create a distance matrix of the points to specify the distance between each point."""
    distance_matrix = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            distance_matrix[i, j] = np.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
    return distance_matrix


def create_population():
    """Create an entire randomly generated population."""
    population = []
    for i in range(n_pop):
        candidate = create_candidate()
        population.append(candidate)
    return population


def create_candidate():
    """Create one candidate solution. Since the first value is always zero, it is implicit."""
    candidate = list(range(1,N))
    random.shuffle(candidate)
    return candidate


def grade_all_solutions(distance_matrix, population):
    distances = []
    for candidate in population:
        distance = grade_solution(distance_matrix, candidate)
        distances.append(distance)
    return distances


def grade_solution(distance_matrix, candidate):
    """Use the candidate and distance matrix to grade an individual candidate."""
    distance = 0
    for i in range(len(candidate)):
        if i == 0:
            distance += distance_matrix[0][candidate[i]]
        elif i == len(candidate):
            distance += distance_matrix[candidate[-1]][0]
        else:
            distance += distance_matrix[candidate[i-1]][candidate[i]]
    return distance


def pick_parent(weights):
    """Picks solutions to become parents.
    Solutions with a better fitness are more likely to be picked."""

    array=np.array(weights)
    temp=array.argsort()
    ranks=np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    score = [len(ranks) - x for x in ranks]
    cum_scores = copy.deepcopy(score)
    for i in range(1,len(cum_scores)):
        cum_scores[i] = score[i] + cum_scores[i-1]
        probs = [x / cum_scores[-1] for x in cum_scores]
        rand = random.random()

    for i in range(0, len(probs)):
        if rand < probs[i]:

            return i


def crossover(p1, p2):
    """Breeds two solutions by selecting a random part of the string and swapping
    the corresponding parts."""
    m = random.randint(0, len(p1) - 1)
    child = unique(p1[:m] + p2[:])
    return child


def mutate(candidate):
    """Mutates a candidate by swapping two points."""
    if random.random() < mutation_prob:
        m = random.randint(0, len(candidate) - 1)
        n = random.randint(0, len(candidate) - 1)
        candidate = swapPositions(candidate, m, n)
    return candidate


def plot_candidate(points, candidate, lowest_distance, generation):
    """Plot a specific candidate solution."""
    # Clear the previous plot

    #plt.figure(figsize=(10,10))
    plt.cla()

    # Set graph settings
    plt.axis([0, 100, 0, 100])
    plt.title(f"Generation {generation}")
    plt.text(5, 5, "Distance = {0:.2f}".format(lowest_distance), fontsize = 12)

    # Cycle through the solution and plot each corresponding line
    for i in range(len(candidate) + 1):
        if i == 0:
            line = [points[0], points[candidate[i]]]
        elif i == len(candidate):
            line = [points[candidate[-1]], points[0]]
        else:
            line = [points[candidate[i-1]], points[candidate[i]]]
        plt.plot(*zip(*line), color='#ffaa00')

    # Plot each point
    plt.scatter(*zip(*points))
    plt.draw()
    plt.pause(0.001)


def unique(sequence):
    """Returns a sequence after removing non unique elements."""
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list


if __name__ == '__main__':
    main()
