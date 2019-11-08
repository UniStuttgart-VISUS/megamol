#include "tsp.h"

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Eigen/Dense"

namespace megamol {
namespace thirdparty {
namespace tsp {

Graph::Graph(const std::vector<Eigen::Vector2f>& points) : num_vertices(points.size()) {
    if (points.size() < 2) {
        throw std::runtime_error("The graph must be constructed from at least two points");
    }

    for (std::size_t outer_index = 0; outer_index < points.size() - 1; ++outer_index) {
        for (std::size_t inner_index = outer_index + 1; inner_index < points.size(); ++inner_index) {
            this->edges[std::make_pair(outer_index, inner_index)] = (points[outer_index] - points[inner_index]).norm();
        }
    }
}

float Graph::get_distance(std::size_t src, std::size_t dest) const {
    return this->edges.find(std::make_pair(std::min(src, dest), std::max(src, dest)))->second;
}

std::size_t Graph::get_num_vertices() const { return this->num_vertices; }

// ----------------------------------------------------------------------------

Genetic::Genetic(const std::shared_ptr<const Graph> graph, const std::size_t size_population,
    const std::size_t generations, const std::size_t mutation_rate)
    : graph(graph), size_population(size_population), generations(generations), mutation_rate(mutation_rate) {}

std::vector<std::size_t> Genetic::run() {
    generate_initial_population();

    for (std::size_t i = 0; i < this->generations; ++i) {
        const auto old_size_population = this->population.size();

        // Select two parents who will participate in the reproduction process
        if (this->population.size() >= 2) {
            if (this->population.size() == 2) {
                cross_over(this->population[0].first, this->population[1].first);
            } else {
                std::size_t parent1, parent2;

                do {
                    // Select two random parents
                    parent1 = std::rand() % this->population.size();
                    parent2 = std::rand() % this->population.size();
                } while (parent1 == parent2);

                cross_over(this->population[parent1].first, this->population[parent2].first);
            }

            // Get difference to check if the population grew
            const auto diff_population = this->population.size() - old_size_population;

            if (diff_population == 2) {
                if (this->population.size() > this->size_population) {
                    // Remove the two worst parents of the population
                    this->population.pop_back();
                    this->population.pop_back();
                }
            } else if (diff_population == 1) {
                if (this->population.size() > this->size_population) {
                    // Remove the worst parent of the population
                    this->population.pop_back();
                }
            }
        } else {
            // Apply crossover within the parent
            cross_over(this->population[0].first, this->population[0].first);

            if (this->population.size() > this->size_population) {
                // Remove the worst parent of the population
                this->population.pop_back();
            }
        }
    }

    return this->population[0].first;
}

void Genetic::generate_initial_population() {
    // Insert vertices, the first one is the initial vertex
    std::vector<std::size_t> parent(this->graph->get_num_vertices());
    std::iota(parent.begin(), parent.end(), 0);

    const auto total_cost = calculate_cost(parent);

    if (total_cost) {
        this->population.push_back(std::make_pair(parent, *total_cost));
    }

    // Generate random permutations
    auto rng = std::default_random_engine{};

    for (std::size_t i = 0; i < generations && this->population.size() != this->size_population; ++i) {
        std::shuffle(
            parent.begin() + 1, parent.begin() + (std::rand() % (this->graph->get_num_vertices() - 1) + 1), rng);

        const auto total_cost = calculate_cost(parent);

        // checks if permutation is a valid solution and if not exists
        if (total_cost && !exists_chromosome(parent)) {
            this->population.push_back(std::make_pair(parent, *total_cost));
        }
    }

    // Checks for a valid population
    if (this->population.size() == 0) {
        throw std::runtime_error("No initial population was generated for the travelling salesman problem");
    }

    std::sort(this->population.begin(), this->population.end(), sort_pred());
}

void Genetic::cross_over(std::vector<std::size_t>& parent1, std::vector<std::size_t>& parent2) {
    std::vector<std::size_t> child1, child2;

    // Map of genes, checks if already selected
    std::map<std::size_t, std::size_t> genes1, genes2;

    for (std::size_t i = 0; i < this->graph->get_num_vertices(); ++i) {
        // Initially the genes not are used
        genes1[parent1[i]] = 0;
        genes2[parent2[i]] = 0;
    }

    // Generate random points
    auto point1 = std::rand() % (this->graph->get_num_vertices() - 3) + 1;
    auto point2 = std::rand() % (this->graph->get_num_vertices() - point1 - 2) + point1 + 1;

    // Adjust the points randomly if they are equal
    if (point1 == point2) {
        const auto decision = std::rand() % 2;

        if (decision == 0) {
            --point1;
        } else {
            ++point2;
        }
    }

    // Generate children

    // Until point1, child1 receives genes of the parent1 and child2 receives genes of the parent2
    for (std::size_t i = 0; i < point1; ++i) {
        // Add and mark genes
        child1.push_back(parent1[i]);
        child2.push_back(parent2[i]);

        genes1[parent1[i]] = 1;
        genes2[parent2[i]] = 1;
    }

    // Mark remaining genes
    for (std::size_t i = point2 + 1; i < this->graph->get_num_vertices(); ++i) {
        genes1[parent1[i]] = 1;
        genes2[parent2[i]] = 1;
    }

    // Here it is inverted: child1 receives genes of the parent2 and child2 receives genes of the parent1
    for (std::size_t i = point2; i >= point1; --i) {
        if (genes1[parent2[i]] == 0) {
            child1.push_back(parent2[i]);

            genes1[parent2[i]] = 1;
        } else {
            // If the gene already is used, chooses gene that is not used
            for (auto it = genes1.begin(); it != genes1.end(); ++it) {
                if (it->second == 0) {
                    child1.push_back(it->first);

                    genes1[it->first] = 1;

                    break;
                }
            }
        }

        if (genes2[parent1[i]] == 0) {
            child2.push_back(parent1[i]);

            genes2[parent1[i]] = 1;
        } else {
            // If the gene already is used, chooses gene that is not used
            for (auto it = genes2.begin(); it != genes2.end(); ++it) {
                if (it->second == 0) {
                    child2.push_back(it->first);

                    genes2[it->first] = 1;

                    break;
                }
            }
        }
    }

    // Remaining genes: child1 receives genes of the parent1 and child2 receives genes of the parent2
    for (std::size_t i = point2 + 1; i < this->graph->get_num_vertices(); ++i) {
        child1.push_back(parent1[i]);
        child2.push_back(parent2[i]);
    }

    // Mutation
    auto mutation = std::rand() % 100 + 1;

    if (mutation <= this->mutation_rate) {
        // Make a mutation: change of two genes
        const auto index_gene1 = std::rand() % (this->graph->get_num_vertices() - 1) + 1;
        const auto index_gene2 = std::rand() % (this->graph->get_num_vertices() - 1) + 1;

        auto aux = child1[index_gene1];
        child1[index_gene1] = child1[index_gene2];
        child1[index_gene2] = aux;

        aux = child2[index_gene1];
        child2[index_gene1] = child2[index_gene2];
        child2[index_gene2] = aux;
    }

    const auto total_cost_child1 = calculate_cost(child1);
    const auto total_cost_child2 = calculate_cost(child2);

    // Check if is a valid solution and not exists in the population
    if (total_cost_child1 && !exists_chromosome(child1)) {
        // Add child in the population
        insert_binary_search(std::move(child1), *total_cost_child1);
    }

    // checks again...
    if (total_cost_child2 && !exists_chromosome(child2)) {
        // Add child in the population
        insert_binary_search(std::move(child2), *total_cost_child2);
    }
}

bool Genetic::exists_chromosome(const std::vector<std::size_t>& v) const {
    for (auto it = this->population.begin(); it != this->population.end(); ++it) {
        const auto& vec = (*it).first;

        if (std::equal(v.begin(), v.end(), vec.begin())) {
            return true;
        }
    }

    return false;
}

std::optional<float> Genetic::calculate_cost(const std::vector<std::size_t>& solution) const {
    auto total_cost = 0.0f;

    // Checks if not contains elements repeated
    std::set<std::size_t> set_solution(solution.begin(), solution.end());

    if (set_solution.size() != this->graph->get_num_vertices()) {
        return std::nullopt;
    }

    // Sum up the cost
    for (std::size_t i = 0; i < this->graph->get_num_vertices(); ++i) {
        total_cost += this->graph->get_distance(solution[i], solution[(i + 1) % this->graph->get_num_vertices()]);
    }

    return total_cost;
}

void Genetic::insert_binary_search(std::vector<std::size_t> child, const float total_cost) {
    std::size_t imin = 0;
    std::size_t imax = this->population.size() - 1;

    if (this->population.size() > 0) {
        while (imax >= imin) {
            const auto imid = (imin + imax) / 2;

            if (total_cost == this->population[imid].second) {
                this->population.insert(this->population.begin() + imid, std::make_pair(child, total_cost));
                return;
            } else if (total_cost > this->population[imid].second) {
                imin = imid + 1;
            } else if (imid > 0) {
                imax = imid - 1;
            } else {
                break;
            }
        }
    }

    this->population.insert(this->population.begin() + imin, std::make_pair(child, total_cost));
}

} // namespace tsp
} // namespace thirdparty
} // namespace megamol
