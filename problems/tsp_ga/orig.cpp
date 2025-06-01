#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;
vector<vector<double>> distanceMatrix;
vector<vector<int>> population;
int population_size;
int number_of_city;
int max_iteration;
int gap;
double solution, percent, mutate_probability, initial_probability,
    select_probability;
vector<double> resultList;

vector<int> generateRandomSolution(int numCities) {
  vector<int> solution(numCities);

  // Fill the solution vector with city indices
  for (int i = 0; i < numCities; ++i) {
    solution[i] = i;
  }

  // Shuffle the solution vector randomly
  random_shuffle(solution.begin(), solution.end());

  return solution;
}

// Initialize solution with Nearest Neighbor
vector<int> nearestNeighbor(int start) {
  int n = distanceMatrix.size();
  vector<bool> visited(n, false);
  vector<int> tour;
  int current = start;

  tour.push_back(current);
  visited[current] = true;

  for (int i = 0; i < n - 1; ++i) {
    double minDist = numeric_limits<double>::max();
    int nearestNeighbor = -1;

    for (int j = 0; j < n; ++j) {
      if (!visited[j] && j != current && distanceMatrix[current][j] < minDist) {
        minDist = distanceMatrix[current][j];
        nearestNeighbor = j;
      }
    }

    if (nearestNeighbor != -1) {
      tour.push_back(nearestNeighbor);
      visited[nearestNeighbor] = true;
      current = nearestNeighbor;
    }
  }

  return tour;
}

// Define fitness function
double objective_function(const vector<int> &tour) {
  double totalDistance = 0.0;
  for (int i = 0; i < tour.size() - 1; ++i) {
    totalDistance += distanceMatrix[tour[i]][tour[i + 1]];
  }
  totalDistance += distanceMatrix[tour[tour.size() - 1]][tour[0]];
  return totalDistance;
}

vector<int> tournamentSelection(const vector<vector<int>> &population,
                                int tournamentSize) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dist(0, population.size() - 1);

  vector<int> bestTour;
  int bestDistance = numeric_limits<int>::max();

  for (int i = 0; i < tournamentSize; ++i) {
    int randomIndex = dist(gen);
    vector<int> candidateTour = population[randomIndex];
    int candidateDistance = objective_function(candidateTour);
    if (candidateDistance < bestDistance) {
      bestTour = candidateTour;
      bestDistance = candidateDistance;
    }
  }

  return bestTour;
}

// Order crossover
pair<vector<int>, vector<int>> order_crossover(vector<int> parent1,
                                               vector<int> parent2) {
  int first = rand() % (number_of_city / 2 - 1) + 1;
  int second = rand() % (number_of_city / 2) + (number_of_city / 2);
  vector<int> offspring1(number_of_city), offspring2(number_of_city);
  for (int i = first; i <= second; i++) {
    offspring1[i] = parent1[i];
    offspring2[i] = parent2[i];
  }
  int it1 = 0, it2 = 0;
  for (int i = 0; i < number_of_city; i++) {
    if (find(offspring1.begin(), offspring1.end(), parent2[i]) ==
            offspring1.end() &&
        it1 < number_of_city) {
      offspring1[it1] = parent2[i];
      it1++;
      if (it1 == first) it1 = second + 1;
    }
    if (find(offspring2.begin(), offspring2.end(), parent1[i]) ==
            offspring2.end() &&
        it2 < number_of_city) {
      offspring2[it2] = parent1[i];
      it2++;
      if (it2 == first) it2 = second + 1;
    }
  }
  return {offspring1, offspring2};
}

// Mutation
vector<int> mutation(vector<int> offspring) {
  int tmp1 = rand() % number_of_city;
  int tmp2 = rand() % number_of_city;
  while (tmp2 == tmp1) {
    tmp2 = rand() % number_of_city;
  }
  vector<int> mutate = offspring;
  swap(mutate[tmp1], mutate[tmp2]);
  return mutate;
}

// 2-opt operator
vector<int> two_opt_swap(vector<int> solution, int i, int j) {
  vector<int> newSolution = solution;
  reverse(newSolution.begin() + i, newSolution.begin() + j + 1);
  return newSolution;
}

// Random 2-opt shuffle
vector<int> two_opt_shuffle(vector<int> solution) {
  int tmp1 = rand() % number_of_city;
  int tmp2 = rand() % number_of_city;
  while (tmp2 < tmp1) {
    tmp2 = rand() % number_of_city;
  }
  vector<int> newSolution = solution;
  random_shuffle(newSolution.begin() + tmp1, newSolution.begin() + tmp2);
  return newSolution;
}

// 2-opt operator hill-climbing
vector<int> two_opt_hill_climb(vector<int> solution) {
  bool improved = true;
  vector<int> best_solution = solution;
  double best_distance = objective_function(solution);

  while (improved) {
    improved = false;
    for (int i = 0; i < number_of_city - 1; ++i) {
      for (int j = i + 1; j < number_of_city; ++j) {
        vector<int> new_solution = two_opt_swap(best_solution, i, j);
        double new_distance = objective_function(new_solution);
        if (new_distance < best_distance) {
          best_distance = new_distance;
          best_solution = new_solution;
          improved = true;
          break;
        }
      }
      if (improved) {
        break;
      }
    }
  }

  return best_solution;
}

void initialize_population() {
  for (int i = 0; i < population_size; i++) {
    vector<int> tmp;
    if ((double)rand() / RAND_MAX < initial_probability) {
      tmp = nearestNeighbor(i);
    } else {
      tmp = generateRandomSolution(number_of_city);
    }
    // if (number_of_city <= 100)
    // {
    //     tmp = two_opt_hill_climb(tmp);
    // }
    // else
    //     tmp = two_opt_shuffle(tmp);
    population.push_back(tmp);
  }
}

vector<vector<int>> generate_new(vector<vector<int>> pop) {
  vector<vector<int>> newPopulation;
  for (int i = 0; i < population_size; i++) {
    vector<int> parent1 = tournamentSelection(population, population_size / 2);
    vector<int> parent2 = tournamentSelection(population, population_size / 2);
    pair<vector<int>, vector<int>> offspring =
        order_crossover(parent1, parent2);
    vector<int> child;
    if (objective_function(parent1) > objective_function(parent2)) {
      child = offspring.first;
    } else
      child = offspring.second;
    child = two_opt_shuffle(child);
    double x = (double)rand() / RAND_MAX;
    // cout << x << endl;
    if (x < mutate_probability) {
      if (number_of_city <= 100) {
        child = two_opt_hill_climb(child);
      } else
        child = two_opt_shuffle(child);
    }
    newPopulation.push_back(child);
  }
  for (int i = 0; i < population_size; i++) {
    if (objective_function(pop[i]) < objective_function(newPopulation[i])) {
      newPopulation[i] = pop[i];
    }
  }
  return newPopulation;
}

double best_fitness() {
  double res = 1.0 * (1e12);
  for (int i = 0; i < population_size; i++) {
    if (res > objective_function(population[i])) {
      res = objective_function(population[i]);
    }
  }
  return res;
}

void geneticAlgorithm() {
  select_probability = 0.7;
  if (number_of_city <= 50) {
    max_iteration = 5e4;
    mutate_probability = 0.01;
    gap = 5000;
    initial_probability = 0.85;
  } else {
    if (number_of_city < 150) {
      max_iteration = 5e4;
      mutate_probability = 0.005;
      gap = 5000;
      initial_probability = 0.9;
    } else {
      if (number_of_city < 300) {
        max_iteration = 7e3;
        mutate_probability = 0.005;
        gap = 3000;
        initial_probability = 1;
      } else {
        max_iteration = 3e3;
        mutate_probability = 0.001;
        gap = 100;
        initial_probability = 1;
      }
    }
  }
  population_size = min(number_of_city / 2, 100);
  initialize_population();
  solution = best_fitness();
  resultList.push_back(solution);
  for (int i = 0; i < max_iteration; i++) {
    population = generate_new(population);
    double res = best_fitness();
    solution = min(res, solution);
    if (solution < resultList.back()) {
      cout << "Iteration " << i << ": " << solution << endl;
    }
    resultList.push_back(solution);
    if (i > gap && resultList[i - gap] == resultList[i]) break;
  }
}

void input() {
  cin >> number_of_city;
  distanceMatrix = vector(number_of_city, vector<double>(number_of_city));
  for (int i = 0; i < number_of_city; i++) {
    for (int j = 0; j < number_of_city; j++) {
      cin >> distanceMatrix[i][j];
    }
  }
}
int main() {
  cout << "Begin solution" << endl;
  srand(time(NULL));
  input();
  geneticAlgorithm();
  cout << solution << endl;
  return 0;
}
