#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "mesh/MeshCalls.h"

#include "glm/glm.hpp"

#include "CGAL/Exact_predicates_inexact_constructions_kernel.h"
#include "CGAL/Polyhedron_3.h"
#include "CGAL/Polyhedron_incremental_builder_3.h"
#include "CGAL/Side_of_triangle_mesh.h"

namespace megamol::thermodyn {
// using vertex_con_t = std::vector<glm::vec3>;
// using index_con_t = std::vector<glm::uvec3>;

template<typename HDS>
class PolyhedronFactory : public CGAL::Modifier_base<HDS> {
public:
    PolyhedronFactory(glm::vec3 const* vertices, uint64_t vert_count, glm::uvec3 const* indices, uint64_t ind_count)
            : vertices_(vertices), vert_count_(vert_count), indices_(indices), ind_count_(ind_count) {}

    void operator()(HDS& hds) {
        CGAL::Polyhedron_incremental_builder_3<HDS> B(hds, true);
        B.begin_surface(vert_count_, ind_count_);
        using Vertex = HDS::Vertex;
        using Point = Vertex::Point;
        for (uint64_t i = 0; i < vert_count_; ++i) {
            auto const& el = vertices_[i];
            B.add_vertex(Point(el.x, el.y, el.z));
        }
        for (uint64_t i = 0; i < ind_count_; ++i) {
            auto const& el = indices_[i];
            B.begin_facet();
            B.add_vertex_to_facet(el.x);
            B.add_vertex_to_facet(el.y);
            B.add_vertex_to_facet(el.z);
            B.end_facet();
        }
        B.end_surface();
    }

private:
    glm::vec3 const* vertices_;

    uint64_t vert_count_;

    glm::uvec3 const* indices_;

    uint64_t ind_count_;
};

class ParticlesInsideMesh : public core::Module {
public:
    using Gt = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Polyhedron = CGAL::Polyhedron_3<Gt>;
    using HalfedgeDS = Polyhedron::HalfedgeDS;

    using Point = Gt::Point_3;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ParticlesInsideMesh";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "ParticlesInsideMesh";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    ParticlesInsideMesh();

    virtual ~ParticlesInsideMesh();

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(mesh::CallMesh& meshes, core::moldyn::MultiParticleDataCall& particles);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot mesh_in_slot_;

    core::CallerSlot parts_in_slot_;

    std::vector<std::vector<float>> icol_data_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    unsigned int frame_id_ = 0;

    uint64_t out_data_hash_ = 0;
};
} // namespace megamol::thermodyn
