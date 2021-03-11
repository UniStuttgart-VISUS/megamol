#pragma once

// CGAL
#include "CGAL/Exact_predicates_exact_constructions_kernel.h"
#include "CGAL/Exact_predicates_inexact_constructions_kernel.h"

#include "CGAL/Delaunay_triangulation_3.h"
#include "CGAL/Fixed_alpha_shape_3.h"
#include "CGAL/Fixed_alpha_shape_cell_base_3.h"
#include "CGAL/Fixed_alpha_shape_vertex_base_3.h"

#include "CGAL/Min_sphere_d.h"
#include "CGAL/Min_sphere_of_points_d_traits_d.h"

#include "CGAL/Triangulation_vertex_base_with_info_3.h"
// END CGAL

namespace megamol::thermodyn {
    using Gt = CGAL::Exact_predicates_inexact_constructions_kernel;

    using Vbt = CGAL::Triangulation_vertex_base_with_info_3<std::size_t, Gt>;
    using Vb = CGAL::Fixed_alpha_shape_vertex_base_3<Gt, Vbt>;
    using Fb = CGAL::Fixed_alpha_shape_cell_base_3<Gt>;
    using Tds = CGAL::Triangulation_data_structure_3<Vb, Fb>;
    using Triangulation_3 = CGAL::Delaunay_triangulation_3<Gt, Tds>;
    using Alpha_shape_3 = CGAL::Fixed_alpha_shape_3<Triangulation_3>;

    using Point_3 = Alpha_shape_3::Point;

    // using Triangle = Alpha_shape_3::Triangle;
    using Triangle = Triangulation_3::Triangle;
    using Facet = Alpha_shape_3::Facet;
    using Vertex = Alpha_shape_3::Vertex_handle;
}
