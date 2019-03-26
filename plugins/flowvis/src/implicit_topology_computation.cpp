#include "stdafx.h"
#include "implicit_topology_computation.h"

#include <chrono>
#include <future>
#include <thread>
#include <utility>

namespace megamol
{
    namespace flowvis
    {
        implicit_topology_computation::implicit_topology_computation(std::vector<float> positions, std::vector<float> vectors, std::vector<float> points,
            std::vector<int> point_ids, std::vector<float> lines, std::vector<int> line_ids) : delaunay(positions)
        {
            // Set input
            std::swap(this->positions, positions);
            std::swap(this->vectors, vectors);

            std::swap(this->points, points);
            std::swap(this->point_ids, point_ids);

            std::swap(this->lines, lines);
            std::swap(this->line_ids, line_ids);

            // Set state
            this->terminate_computation = false;

            // Compute initial fields
            unsigned int num = this->positions.size() / 2;

            this->labels.resize(num);
            this->distances.resize(num);

            auto calc_dot = [](const float x_1, const float y_1, const float x_2, const float y_2) { return x_1 * x_2 + y_1 * y_2; };
            auto calc_norm = [calc_dot](const float x, const float y) { return calc_dot(x, y, x, y); };
            auto calc_length = [calc_norm](const float x_1, const float y_1, const float x_2, const float y_2) { return std::sqrt(calc_norm(x_1 - x_2, y_1 - y_2)); };

            for (unsigned int n = 0; n < num; ++n)
            {
                const float x_pos = this->positions[n * 2 + 0];
                const float y_pos = this->positions[n * 2 + 1];

                this->distances[n] = std::numeric_limits<float>::max();

                for (unsigned int i = 0; i < this->point_ids.size(); ++i)
                {
                    const float point_x_pos = this->points[i * 2 + 0];
                    const float point_y_pos = this->points[i * 2 + 1];

                    const float distance = calc_length(x_pos, y_pos, point_x_pos, point_y_pos);

                    if (this->distances[n] > distance)
                    {
                        this->labels[n] = static_cast<GLfloat>(this->point_ids[i]);
                        this->distances[n] = distance;
                    }
                }

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

                    if (this->distances[n] > distance)
                    {
                        this->labels[n] = static_cast<GLfloat>(this->line_ids[i]);
                        this->distances[n] = distance;
                    }
                }
            }
        }

        void implicit_topology_computation::start()
        {
            if (!this->terminate_computation)
            {
                // Create promise and get future results
                std::promise<result> promise;
                this->current_result = promise.get_future().share();

                // Start main thread
                std::thread([this](std::promise<result>&& promise)
                {
                    bool finished = false;

                    while (!this->terminate_computation && !finished)
                    {
                        // Produce results...

                        // TODO

                        finished = true;

                        // Set (intermediate) results
                        result current_result;

                        auto mesh = this->delaunay.export_grid();
                        current_result.vertices = mesh.first;
                        current_result.indices = mesh.second;

                        current_result.labels = std::make_shared<std::vector<float>>(this->labels);
                        current_result.distances = std::make_shared<std::vector<float>>(this->distances);

                        current_result.finished = this->terminate_computation || finished;

                        promise.set_value(std::move(current_result));

                        // Prepare new results
                        if (!this->terminate_computation && !finished)
                        {
                            std::swap(promise, std::promise<result>());

                            this->current_result = promise.get_future().share();
                        }

                        // Prevent blocking of CPU
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }

                    this->terminate_computation = false;
                }
                , std::move(promise)).detach();
            }
        }

        void implicit_topology_computation::terminate()
        {
            this->terminate_computation = true;
        }

        std::shared_future<implicit_topology_computation::result> implicit_topology_computation::get_results()
        {
            return this->current_result;
        }
    }
}
