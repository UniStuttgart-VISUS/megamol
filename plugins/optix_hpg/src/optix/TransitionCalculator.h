#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "geometry_calls//MultiParticleDataCall.h"

#include "mesh/MeshCalls.h"

#include "cuda.h"
#include "optix.h"

#include "optix/Utils.h"

#include "MMOptixModule.h"
#include "SBT.h"

#include "optix/Context.h"

namespace megamol::optix_hpg {
class TransitionCalculator : public core::Module {
public:
    std::vector<std::string> requested_lifetime_resources() override {
        auto res = Module::requested_lifetime_resources();
        res.push_back(frontend_resources::CUDA_Context_Req_Name);
        return res;
    }

    static const char* ClassName(void) {
        return "TransitionCalculator";
    }

    static const char* Description(void) {
        return "TransitionCalculator for OptiX";
    }

    static bool IsAvailable(void) {
        return true;
    }

    TransitionCalculator();

    virtual ~TransitionCalculator();

protected:
    bool create() override;

    void release() override;

private:
    bool init();

    bool assertData(mesh::CallMesh& mesh, geocalls::MultiParticleDataCall& particles, unsigned int frameID);

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    core::CalleeSlot out_transitions_slot_;

    core::CallerSlot in_mesh_slot_;

    core::CallerSlot in_paths_slot_;

    OptixModule builtin_triangle_intersector_;

    MMOptixModule mesh_module_;

    MMOptixModule raygen_module_;

    MMOptixModule miss_module_;

    std::vector<CUdeviceptr> ray_buffer_;

    OptixPipeline pipeline_;

    MMOptixSBT sbt_;

    unsigned int _frame_id = std::numeric_limits<unsigned int>::max();

    std::size_t _data_hash = std::numeric_limits<std::size_t>::max();

    std::unique_ptr<Context> optix_ctx_;
};
} // namespace megamol::optix_hpg
