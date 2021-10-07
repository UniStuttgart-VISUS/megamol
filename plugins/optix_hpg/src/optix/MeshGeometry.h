#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mesh/MeshCalls.h"

#include "cuda.h"

#include "SBT.h"

#include "mesh.h"

#include "MMOptixModule.h"

#include "optix/Context.h"

namespace megamol::optix_hpg {
class MeshGeometry : public core::Module {
public:
    static const char* ClassName(void) {
        return "MeshGeometry";
    }

    static const char* Description(void) {
        return "Mesh Geometry for OptiX";
    }

    static bool IsAvailable(void) {
        return true;
    }

    MeshGeometry();

    virtual ~MeshGeometry();

protected:
    bool create() override;

    void release() override;

private:
    void init(Context const& ctx);

    bool assertData(mesh::CallMesh& call, Context const& ctx);

    bool get_data_cb(core::Call& c);

    bool get_extents_cb(core::Call& c);

    core::CalleeSlot _out_geo_slot;

    core::CallerSlot _in_data_slot;

    MMOptixModule mesh_module_;

    MMOptixModule mesh_occlusion_module_;

    OptixModule triangle_intersector_;

    std::vector<SBTRecord<device::MeshGeoData>> sbt_records_;

    std::array<OptixProgramGroup, 2> program_groups_;

    CUdeviceptr _geo_buffer = 0;

    std::vector<CUdeviceptr> mesh_pos_data_;

    std::vector<CUdeviceptr> mesh_idx_data_;

    OptixTraversableHandle _geo_handle;

    unsigned int _frame_id = std::numeric_limits<unsigned int>::max();

    std::size_t _data_hash = std::numeric_limits<std::size_t>::max();

    uint64_t sbt_version = 0;

    uint64_t program_version = 0;
};
} // namespace megamol::optix_hpg
