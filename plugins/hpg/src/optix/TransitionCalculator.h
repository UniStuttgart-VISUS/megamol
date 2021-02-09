#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "mesh/MeshCalls.h"

#include "hpg/optix/CallContext.h"

#include "cuda.h"
#include "optix.h"

#include "hpg/optix/Utils.h"

#include "MMOptixModule.h"
#include "SBT.h"

namespace megamol::hpg::optix {
class TransitionCalculator : public core::Module {
public:
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
    bool init(CallContext& ctx);

    bool assertData(
        CallContext& ctx, mesh::CallMesh& mesh, core::moldyn::MultiParticleDataCall& particles, unsigned int frameID);

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    core::CalleeSlot out_transitions_slot_;

    core::CallerSlot in_mesh_slot_;

    core::CallerSlot in_paths_slot_;

    core::CallerSlot in_ctx_slot_;

    OptixModule builtin_triangle_intersector_;

    MMOptixModule mesh_module_;

    MMOptixModule raygen_module_;

    MMOptixModule miss_module_;

    std::vector<CUdeviceptr> ray_buffer_;

    OptixPipeline pipeline_;

    MMOptixSBT sbt_;

    unsigned int _frame_id = std::numeric_limits<unsigned int>::max();

    std::size_t _data_hash = std::numeric_limits<std::size_t>::max();
};
} // namespace megamol::hpg::optix
