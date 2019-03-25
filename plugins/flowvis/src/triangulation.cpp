#include "stdafx.h"
#include "triangulation.h"

#include "glad/glad.h"

#include <utility>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        triangulation::triangulation(const std::vector<GLfloat>& initial_points) : point_index(0)
        {
            if (!initial_points.empty())
            {
                insert_points(initial_points);
            }
        }

        void triangulation::insert_points(const std::vector<GLfloat>& new_points)
        {
            // Get new points and add them to the triangulation and the output
            std::vector<std::pair<point_t, std::size_t>> points;
            points.reserve(new_points.size() / 2);

            for (std::size_t i = 0; i < new_points.size() / 2; ++i)
            {
                points.push_back(std::make_pair(point_t(static_cast<double>(new_points[i * 2 + 0]), static_cast<double>(new_points[i * 2 + 1])), this->point_index++));
            }

            // Apply delaunay
            this->delaunay.insert(points.begin(), points.end());
        }

        std::pair<std::shared_ptr<std::vector<GLfloat>>, std::shared_ptr<std::vector<GLuint>>> triangulation::export_grid() const
        {
            // Extract points
            auto vertices = std::make_shared<std::vector<GLfloat>>(this->delaunay.number_of_vertices() * 2);

            for (auto vertex_it = this->delaunay.finite_vertices_begin(); vertex_it != this->delaunay.finite_vertices_end(); ++vertex_it)
            {
                (*vertices)[vertex_it->info() * 2 + 0] = static_cast<GLfloat>(CGAL::to_double(vertex_it->point()[0]));
                (*vertices)[vertex_it->info() * 2 + 1] = static_cast<GLfloat>(CGAL::to_double(vertex_it->point()[1]));
            }

            // Extract cells
            auto indices = std::make_shared<std::vector<GLuint>>(get_number_of_cells() * 3);

            std::size_t cell_index = 0;

            for (auto face_it = get_finite_cells_begin(); face_it != get_finite_cells_end(); ++face_it)
            {
                // Get all triangle points
                for (unsigned int i = 0; i < 3; ++i)
                {
                    (*indices)[cell_index++] = static_cast<GLuint>(face_it->vertex(i)->info());
                }
            }

            return std::make_pair(vertices, indices);
        }

        std::vector<triangulation::vertex_t> triangulation::get_neighbors(const vertex_t& vertex) const
        {
            std::vector<vertex_t> neighbors;

            auto circulator = this->delaunay.incident_vertices(vertex);
            const auto first = circulator;

            do
            {
                if (!this->delaunay.is_infinite(circulator->handle()))
                {
                    neighbors.push_back(circulator->handle());
                }
            } while (++circulator != first);

            return neighbors;
        }

        std::size_t triangulation::get_number_of_cells() const
        {
            return this->delaunay.number_of_faces();
        }

        triangulation::cell_it_t triangulation::get_finite_cells_begin() const
        {
            return this->delaunay.finite_faces_begin();
        }

        triangulation::cell_it_t triangulation::get_finite_cells_end() const
        {
            return this->delaunay.finite_faces_end();
        }

        void triangulation::to_c_point(const point_t& point, float* c_point) const
        {
            c_point[0] = static_cast<float>(CGAL::to_double(point[0]));
            c_point[1] = static_cast<float>(CGAL::to_double(point[1]));
        }
    }
}