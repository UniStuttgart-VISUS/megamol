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
            /**
            * Constructor
            *
            * @param initial_points Initial points to add to the triangulation
            */
            triangulation(const std::vector<GLfloat>& initial_points);

            /**
            * Iteratively insert points for triangulation
            *
            * @param new_points New points to add to the triangulation
            */
            void insert_points(const std::vector<GLfloat>& new_points);

            /**
            * Export triangulation as grid
            *
            * @param grid Grid storing the triangulation
            */
            std::pair<std::shared_ptr<std::vector<GLfloat>>, std::shared_ptr<std::vector<GLuint>>> export_grid() const;

            /**
            * Get neighbor vertices
            *
            * @param vertex Input vertex
            *
            * @return Neighbors of the input vertex
            */
            std::vector<vertex_t> get_neighbors(const vertex_t& vertex) const;

            /**
            * Get number of cells in triangulation
            *
            * @return Number of cells
            */
            std::size_t get_number_of_cells() const;

            /**
            * Get cell iterator of triangulation
            *
            * @return Cell iterator
            */
            cell_it_t get_finite_cells_begin() const;

            /**
            * Get past-the-end cell iterator of triangulation
            *
            * @return Cell iterator
            */
            cell_it_t get_finite_cells_end() const;

            /**
            * Convert CGAL point to c-style array
            *
            * @param point CGAL point
            * @param c_point C-style array
            */
            void to_c_point(const point_t& point, float* c_point) const;

        private:
            // Counter for mapping triangulated points to original input
            std::size_t point_index;

        public:
            // Access to delaunay triangulation
            delaunay_t delaunay;
        };
    }
}
