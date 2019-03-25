#pragma once

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_face_base_2.h>
#include <CGAL/Triangulation_data_structure_2.h>

#include "glad/glad.h"

#include <utility>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        class triangulation
        {
        private:
            typedef CGAL::Exact_predicates_inexact_constructions_kernel kernel;

            typedef CGAL::Triangulation_vertex_base_with_info_2<std::size_t, kernel> vertex_base;
            typedef CGAL::Triangulation_face_base_2<kernel> face_base;

            typedef CGAL::Triangulation_data_structure_2<vertex_base, face_base> data_structure;

        public:
            typedef CGAL::Delaunay_triangulation_2<kernel, data_structure> delaunay_t;

            typedef delaunay_t::Point point_t;
            typedef delaunay_t::Finite_faces_iterator cell_it_t;

            typedef delaunay_t::Vertex_handle vertex_t;

        public:
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="initial_points">Initial points to add to the triangulation</param>
            triangulation(const std::vector<GLfloat>& initial_points);

            /// <summary>
            /// Iteratively insert points for triangulation
            /// </summary>
            /// <param name="new_points">New points to add to the triangulation</param>
            void insert_points(const std::vector<GLfloat>& new_points);

            /// <summary>
            /// Export triangulation as grid
            /// </summary>
            /// <param name="grid">Grid storing the triangulation</param>
            std::pair<std::shared_ptr<std::vector<GLfloat>>, std::shared_ptr<std::vector<GLuint>>> export_grid() const;

            /// <summary>
            /// Get neighbor vertices
            /// </summary>
            /// <param name="vertex">Input vertex</param>
            /// <returns>Neighbors of the input vertex</returns>
            std::vector<vertex_t> get_neighbors(const vertex_t& vertex) const;

            /// <summary>
            /// Get number of cells in triangulation
            /// </summary>
            /// <returns>Number of cells</returns>
            std::size_t get_number_of_cells() const;

            /// <summary>
            /// Get cell iterator of triangulation
            /// </summary>
            /// <returns>Cell iterator</returns>
            cell_it_t get_finite_cells_begin() const;

            /// <summary>
            /// Get past-the-end cell iterator of triangulation
            /// </summary>
            /// <returns>Cell iterator</returns>
            cell_it_t get_finite_cells_end() const;

            /// <summary>
            /// Convert CGAL point to c-style array
            /// </summary>
            /// <param name="point">CGAL point</param>
            /// <param name="c_point">C-style array</param>
            void to_c_point(const point_t& point, float* c_point) const;

        private:
            /// point_t counter for mapping triangulated points to original input
            std::size_t point_index;

        public:
            /// Access to delaunay triangulation
            delaunay_t delaunay;
        };
    }
}
