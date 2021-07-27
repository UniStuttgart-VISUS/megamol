#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "mesh/MeshCalls.h"

#include "glm/glm.hpp"

namespace megamol::thermodyn {
class ParticleSurfaceRefinement : public core::Module {
public:
    using vertex_con_t = std::vector<glm::vec3>;
    using normals_con_t = std::vector<glm::vec3>;
    using index_con_t = std::vector<uint32_t>;

    using idx_map_t = std::unordered_map<uint64_t /* point idx */, uint64_t /* vertex idx */>;

    /** Return module class name */
    static const char* ClassName(void) {
        return "ParticleSurfaceRefinement";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "ParticleSurfaceRefinement";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    ParticleSurfaceRefinement();

    virtual ~ParticleSurfaceRefinement();

protected:
    bool create() override;

    void release() override;

private:
    bool is_dirty() const {
        return volume_thres_slot_.IsDirty() || area_thres_slot_.IsDirty();
    }

    void reset_dirty() {
        volume_thres_slot_.ResetDirty();
        area_thres_slot_.ResetDirty();
    }

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    bool assert_data(mesh::CallMesh& meshes);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot data_in_slot_;

    core::param::ParamSlot volume_thres_slot_;

    core::param::ParamSlot area_thres_slot_;

    std::vector<vertex_con_t> vertices_;

    std::vector<normals_con_t> normals_;

    std::vector<index_con_t> indices_;

    std::shared_ptr<mesh::MeshDataAccessCollection> mesh_col_;

    uint64_t version_ = 0;
};
} // namespace megamol::thermodyn
