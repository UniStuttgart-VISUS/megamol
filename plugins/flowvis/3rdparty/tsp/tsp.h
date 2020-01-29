/***
 *
 * This source code uses a genetic algorithm to approximate a solution of
 * the travelling salesman problem.
 *
 * This code originates from https://github.com/marcoscastro/tsp_genetic
 * by Marcos Castro de Souza.
 *
 */
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "Eigen/Dense"

namespace megamol {
namespace thirdparty {
namespace tsp {

// class that represents the graph
class Graph {
public:
    /**
     * Create a fully connected graph from a set of points, assigning as edge
     * weights their Euclidean distance
     */
    Graph(const std::vector<Eigen::Vector2f>& points);

    /**
     * Answer the Euclidean distance between to points
     */
    float get_distance(std::size_t src, std::size_t dest) const;

    /**
     * Answer the number of vertices in the graph
     */
    std::size_t get_num_vertices() const;

private:
    // Number of vertices stored in the graph
    std::size_t num_vertices;

    // Edges with their respective weight
    std::map<std::pair<std::size_t, std::size_t>, float> edges;
};

// class that represents genetic algorithm
class Genetic {
private:
    typedef std::pair<std::vector<std::size_t>, float> my_pair;

    // Sort vector with pair
    struct sort_pred {
        bool operator()(const my_pair& firstElem, const my_pair& secondElem) {
            return firstElem.second < secondElem.second;
        }
    };

public:
    /**
     * Setup algorithm, providing the graph and parameters for the genetic algorithm
     */
    Genetic(std::shared_ptr<const Graph> graph, std::size_t size_population, std::size_t generations,
        std::size_t mutation_rate);

    /**
     * Run the algorithm and return the found near-optimal solution
     */
    std::vector<std::size_t> run();

private:
    /**
     * Generate the initial population
     */
    void generate_initial_population();

    /**
     * Makes the crossover
     * This crossover selects two random points
     * These points generates substrings in both parents
     * The substring inverted of parent1 is placed in parent2 and
     * the substring inverted of parent2 is placed in parent1
     * 
     * Example:
     *     parent1: 1 2 3 4 5
     *     parent2: 1 2 4 5 3
     * 
     *     substring in parent1: 2 3 4
     *     substring in parent2: 2 4 5
     * 
     *     substring inverted in parent1: 4 3 2
     *     substring inverted in parent2: 5 4 2
     * 
     *     child1: 1 5 4 2 5
     *     child2: 1 4 3 2 3
     * 
     *     Children are invalids: 5 appears 2x in child1 and 3 appears 2x in child2
     *     Solution: map of genes that checks if genes are not used
     */
    void cross_over(std::vector<std::size_t>& parent1, std::vector<std::size_t>& parent2);

    /**
     * Check for the existance of the given chromosome
     */
    bool exists_chromosome(const std::vector<std::size_t>& v) const;

    /**
     * Calculate the cost for the given possible solution
     */
    std::optional<float> calculate_cost(const std::vector<std::size_t>& solution) const;

    /**
     * Use binary search for insertion
     */
    void insert_binary_search(std::vector<std::size_t> child, float total_cost);

    // The graph
    const std::shared_ptr<const Graph> graph;

    // Population of possible solutions with their respective cost
    std::vector<my_pair> population;

    // Parameters for the genetic algorithm
    std::size_t size_population;
    std::size_t generations;
    std::size_t mutation_rate;
};

} // namespace tsp
} // namespace thirdparty
} // namespace megamol
