#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

#include "mesh/MeshCalls.h"

#include "CGAL/Exact_predicates_inexact_constructions_kernel.h"

#include "CGAL/Delaunay_triangulation_3.h"
#include "CGAL/Fixed_alpha_shape_3.h"
#include "CGAL/Fixed_alpha_shape_cell_base_3.h"
#include "CGAL/Fixed_alpha_shape_vertex_base_3.h"

#include "CGAL/Triangulation_vertex_base_with_info_3.h"

namespace megamol::thermodyn {
class ParticleSurface2 : public core::Module {
public:
    using Gt = CGAL::Exact_predicates_inexact_constructions_kernel;

    using Vbt = CGAL::Triangulation_vertex_base_with_info_3<uint64_t, Gt>;
    using Vb = CGAL::Fixed_alpha_shape_vertex_base_3<Gt, Vbt>;
    using Fb = CGAL::Fixed_alpha_shape_cell_base_3<Gt>;
    using Tds = CGAL::Triangulation_data_structure_3<Vb, Fb>;
    using Triangulation_3 = CGAL::Delaunay_triangulation_3<Gt, Tds>;
    using Alpha_shape_3 = CGAL::Fixed_alpha_shape_3<Triangulation_3>;

    using Point_3 = Alpha_shape_3::Point;
    using Facet = Alpha_shape_3::Facet;

    using point_w_info_t = std::pair<Point_3, uint64_t>;

    using vertex_con_t = std::vector<glm::vec3>;
    using normals_con_t = std::vector<glm::vec3>;
    using index_con_t = std::vector<uint32_t>;

    using idx_map_t = std::unordered_map<uint64_t /* point idx */, uint64_t /* vertex idx */>;

    /** Return module class name */
    static const char* ClassName(void) {
        return "ParticleSurface2";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Determines the surface of MD data";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    ParticleSurface2();

    virtual ~ParticleSurface2();

protected:
    bool create() override;

    void release() override;

private:
    bool is_dirty() const {
        return alpha_slot_.IsDirty();
    }

    void reset_dirty() {
        alpha_slot_.ResetDirty();
    }

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(core::moldyn::MultiParticleDataCall& particles);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot data_in_slot_;

    core::param::ParamSlot alpha_slot_;

    std::shared_ptr<mesh::MeshDataAccessCollection> mesh_col_;

    std::vector<vertex_con_t> vertices_;

    std::vector<normals_con_t> normals_;

    std::vector<index_con_t> indices_;

    std::vector<index_con_t> part_indices_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    unsigned int frame_id_ = 0;

    uint64_t version = 0;
};
} // namespace megamol::thermodyn
