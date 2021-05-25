#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"

#include "CUDA_Context.h"

#include "mesh/MeshCalls.h"

#include "cuda.h"
#include "optix.h"

#include "optix/Utils.h"

#include "MMOptixModule.h"
#include "SBT.h"

#include "optix/Context.h"

namespace megamol::optix_hpg {
class ParticleMeshDistance : public core::Module {
public:
    std::vector<std::string> requested_lifetime_resources() override {
        auto res = Module::requested_lifetime_resources();
        res.push_back(frontend_resources::CUDA_Context_Req_Name);
        return res;
    }

    static const char* ClassName(void) {
        return "ParticleMeshDistance";
    }

    static const char* Description(void) {
        return "ParticleMeshDistance for OptiX";
    }

    static bool IsAvailable(void) {
        return true;
    }

    ParticleMeshDistance();

    virtual ~ParticleMeshDistance();

protected:
    bool create() override;

    void release() override;

private:
    bool init();

    bool assertData(mesh::CallMesh& mesh, core::moldyn::MultiParticleDataCall& particles, unsigned int frameID);

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    core::CalleeSlot out_stats_slot_;

    core::CallerSlot in_data_slot_;

    core::CallerSlot in_mesh_slot_;

    core::param::ParamSlot frame_skip_slot_;

    std::unique_ptr<Context> optix_ctx_;

    OptixModule builtin_triangle_intersector_;

    MMOptixModule mesh_module_;

    MMOptixModule raygen_module_;

    MMOptixModule miss_module_;

    //std::vector<CUdeviceptr> ray_buffer_;

    OptixPipeline pipeline_;

    MMOptixSBT sbt_;

    std::shared_ptr<mesh::MeshDataAccessCollection> mesh_access_collection_;

    std::vector<std::vector<float>> part_mesh_distances_;

    std::vector<std::pair<float, float>> pmd_minmax_;

    std::vector<std::vector<float>> positions_;

    unsigned int frame_id_ = -1;

    std::uint64_t out_data_hash_ = 0;
};
} // namespace megamol::optix_hpg
