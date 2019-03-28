#include "stdafx.h"
#include "implicit_topology_computation.h"

#include "../cuda/streamlines.h"

#include <chrono>
#include <future>
#include <thread>
#include <utility>

namespace megamol
{
    namespace flowvis
    {
        implicit_topology_computation::implicit_topology_computation(std::array<int, 2> resolution, std::array<float, 4> domain,
            std::vector<float> positions, std::vector<float> vectors, std::vector<float> points, std::vector<int> point_ids,
            std::vector<float> lines, std::vector<int> line_ids, const float integration_timestep, const float max_integration_error)
            : resolution(std::move(resolution)), domain(std::move(domain)), positions(std::move(positions)),
            vectors(std::move(vectors)), points(std::move(points)), point_ids(std::move(point_ids)), lines(std::move(lines)),
            line_ids(std::move(line_ids)), integration_timestep(integration_timestep), max_integration_error(max_integration_error),
            num_integration_steps_performed(0), terminate_computation(false)
        {
            // Store positions
            this->positions_forward = this->positions;
            this->positions_backward = this->positions;

            // Compute initial fields
            unsigned int num = this->resolution[0] * this->resolution[1];

            this->labels_forward.resize(num);
            this->distances_forward.resize(num);
            this->terminations_forward.resize(num);

            this->labels_backward.resize(num);
            this->distances_backward.resize(num);
            this->terminations_backward.resize(num);

            auto calc_dot = [](const float x_1, const float y_1, const float x_2, const float y_2) { return x_1 * x_2 + y_1 * y_2; };
            auto calc_norm = [calc_dot](const float x, const float y) { return calc_dot(x, y, x, y); };
            auto calc_length = [calc_norm](const float x_1, const float y_1, const float x_2, const float y_2) { return std::sqrt(calc_norm(x_1 - x_2, y_1 - y_2)); };

            for (unsigned int n = 0; n < num; ++n)
            {
                const float x_pos = this->positions[n * 2 + 0];
                const float y_pos = this->positions[n * 2 + 1];

                this->distances_forward[n] = this->distances_backward[n] = std::numeric_limits<float>::max();
                this->terminations_forward[n] = this->terminations_backward[n] = 0.0f;

                // Compute minimum distance to convergence structures represented by points
                for (unsigned int i = 0; i < this->point_ids.size(); ++i)
                {
                    const float point_x_pos = this->points[i * 2 + 0];
                    const float point_y_pos = this->points[i * 2 + 1];

                    const float distance = calc_length(x_pos, y_pos, point_x_pos, point_y_pos);

                    if (this->distances_forward[n] > distance)
                    {
                        this->labels_forward[n] = this->labels_backward[n] = static_cast<GLfloat>(this->point_ids[i]);
                        this->distances_forward[n] = this->distances_backward[n] = distance;
                    }
                }

                // Compute minimum distance to convergence structures represented by lines
                for (unsigned int i = 0; i < this->line_ids.size(); ++i)
                {
                    float distance = 0.0f;

                    const float line_1_x_pos = this->lines[i * 4 + 0];
                    const float line_1_y_pos = this->lines[i * 4 + 1];
                    const float line_2_x_pos = this->lines[i * 4 + 2];
                    const float line_2_y_pos = this->lines[i * 4 + 3];

                    const float length = calc_length(line_1_x_pos, line_1_y_pos, line_2_x_pos, line_2_y_pos);

                    const float line_vector_x = length == 0.0f ? 0.0f : (line_2_x_pos - line_1_x_pos) / length;
                    const float line_vector_y = length == 0.0f ? 0.0f : (line_2_y_pos - line_1_y_pos) / length;

                    if (line_vector_x == 0.0f && line_vector_y == 0.0f)
                    {
                        distance = calc_length(x_pos, y_pos, line_1_x_pos, line_1_y_pos);
                    }
                    else
                    {
                        const float point_distance = calc_dot(line_vector_x, line_vector_y, x_pos - line_1_x_pos, y_pos - line_1_y_pos);

                        if (point_distance < 0.0f)
                        {
                            distance = calc_length(x_pos, y_pos, line_1_x_pos, line_1_y_pos);
                        }
                        else if (point_distance > length)
                        {
                            distance = calc_length(x_pos, y_pos, line_2_x_pos, line_2_y_pos);
                        }
                        else
                        {
                            const float projection_x_pos = line_1_x_pos + line_vector_x * point_distance;
                            const float projection_y_pos = line_1_y_pos + line_vector_y * point_distance;

                            distance = calc_length(x_pos, y_pos, projection_x_pos, projection_y_pos);
                        }
                    }

                    if (this->distances_forward[n] > distance)
                    {
                        this->labels_forward[n] = this->labels_backward[n] = static_cast<GLfloat>(this->line_ids[i]);
                        this->distances_forward[n] = this->distances_backward[n] = distance;
                    }
                }
            }

            // Initialize triangulation
            this->delaunay.insert_points(this->positions);
        }

        implicit_topology_computation::implicit_topology_computation(std::array<int, 2> resolution, std::array<float, 4> domain,
            std::vector<float> positions, std::vector<float> vectors, std::vector<float> points, std::vector<int> point_ids,
            std::vector<float> lines, std::vector<int> line_ids, result previous_result)
            : resolution(std::move(resolution)), domain(std::move(domain)), positions(std::move(positions)),
            vectors(std::move(vectors)), points(std::move(points)), point_ids(std::move(point_ids)), lines(std::move(lines)),
            line_ids(std::move(line_ids)), integration_timestep(previous_result.computation_state.integration_timestep),
            max_integration_error(previous_result.computation_state.max_integration_error),
            num_integration_steps_performed(previous_result.computation_state.num_integration_steps), terminate_computation(false),
            positions_forward(*previous_result.positions_forward), positions_backward(*previous_result.positions_backward),
            labels_forward(*previous_result.labels_forward), distances_forward(*previous_result.distances_forward),
            terminations_forward(*previous_result.terminations_forward), labels_backward(*previous_result.labels_backward),
            distances_backward(*previous_result.distances_backward), terminations_backward(*previous_result.terminations_backward),
            delaunay(*previous_result.vertices)
        {
        }

        implicit_topology_computation::~implicit_topology_computation()
        {
            terminate();
        }

        void implicit_topology_computation::start(const unsigned int num_integration_steps,
            const float refinement_threshold, const bool refine_at_labels, const float distance_difference_threshold,
            const unsigned int num_particles_per_batch, const unsigned int num_integration_steps_per_batch)
        {
            // Prepare results
            {
                std::promise<result> promise;
                this->current_result = promise.get_future().share();

                // Set initial result
                set_result(promise, num_integration_steps <= this->num_integration_steps_performed);

                // Check if there is actually something to do
                if (num_integration_steps <= this->num_integration_steps_performed)
                {
                    return;
                }
            }

            // Start computation
            std::promise<result> promise;
            this->current_result = promise.get_future().share();

            if (this->computation.joinable())
            {
                this->computation.join();
            }

            this->computation = std::thread(&implicit_topology_computation::run, this, std::move(promise),
                num_integration_steps, refinement_threshold, refine_at_labels, distance_difference_threshold,
                num_particles_per_batch, num_integration_steps_per_batch);
        }

        void implicit_topology_computation::terminate()
        {
            this->terminate_computation = true;

            if (this->computation.joinable())
            {
                this->computation.join();
            }

            this->terminate_computation = false;
        }

        std::shared_future<implicit_topology_computation::result> implicit_topology_computation::get_results()
        {
            return this->current_result;
        }

        void implicit_topology_computation::run(std::promise<result>&& promise, const unsigned int num_integration_steps,
            const float refinement_threshold, const bool refine_at_labels, const float distance_difference_threshold,
            const unsigned int num_particles_per_batch, const unsigned int num_integration_steps_per_batch)
        {
            streamlines_cuda streamlines(this->resolution, this->domain, this->vectors, this->points, this->point_ids,
                this->lines, this->line_ids, this->integration_timestep, this->max_integration_error);

            bool finished = false;
            bool finished_integration = false;
   
            auto positions_forward = this->positions_forward;
            auto positions_backward = this->positions_backward;

            while (!finished && !this->terminate_computation)
            {
                // Integrate stream lines
                {
                    const unsigned int num_steps = std::min(num_integration_steps - this->num_integration_steps_performed, num_integration_steps_per_batch);

                    streamlines.update_labels(positions_forward, this->labels_forward, this->distances_forward, this->terminations_forward,
                        num_steps, 1.0f, num_particles_per_batch);

                    streamlines.update_labels(positions_backward, this->labels_backward, this->distances_backward, this->terminations_backward,
                        num_steps, -1.0f, num_particles_per_batch);

                    this->num_integration_steps_performed += num_steps;

                    // Check if all integration steps have been performed
                    if (this->num_integration_steps_performed >= num_integration_steps)
                    {
                        finished_integration = true;
                    }
                }

                // Refine grid
                if (finished_integration)
                {
                    // TODO

                    // Check if new points have been added; if not, the computation is finished
                    if (true) // TODO
                    {
                        finished = true;
                    }

                    finished_integration = false;
                }

                // Set (intermediate) results
                set_result(promise, finished || this->terminate_computation);

                // Prepare new results
                if (!finished)
                {
                    std::swap(promise, std::promise<result>());

                    this->current_result = promise.get_future().share();
                }

                // Prevent blocking of CPU
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        void implicit_topology_computation::set_result(std::promise<result>& promise, const bool finished)
        {
            result current_result;

            auto mesh = this->delaunay.export_grid();
            current_result.vertices = mesh.first;
            current_result.indices = mesh.second;

            current_result.positions_forward = std::make_shared<std::vector<float>>(this->positions_forward);
            current_result.labels_forward = std::make_shared<std::vector<float>>(this->labels_forward);
            current_result.distances_forward = std::make_shared<std::vector<float>>(this->distances_forward);
            current_result.terminations_forward = std::make_shared<std::vector<float>>(this->terminations_forward);

            current_result.positions_backward = std::make_shared<std::vector<float>>(this->positions_backward);
            current_result.labels_backward = std::make_shared<std::vector<float>>(this->labels_backward);
            current_result.distances_backward = std::make_shared<std::vector<float>>(this->distances_backward);
            current_result.terminations_backward = std::make_shared<std::vector<float>>(this->terminations_backward);

            current_result.finished = finished;

            current_result.computation_state.integration_timestep = this->integration_timestep;
            current_result.computation_state.max_integration_error = this->max_integration_error;
            current_result.computation_state.num_integration_steps = this->num_integration_steps_performed;

            promise.set_value(std::move(current_result));
        }
    }
}
