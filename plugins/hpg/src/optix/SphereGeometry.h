#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "hpg/optix/CallContext.h"

#include "cuda.h"

#include "SBT.h"

#include "sphere.h"

#include "MMOptixModule.h"

namespace megamol::hpg::optix {
class SphereGeometry : public core::Module {
public:
    static const char* ClassName(void) {
        return "SphereGeometry";
    }

    static const char* Description(void) {
        return "Sphere Geometry for OptiX";
    }

    static bool IsAvailable(void) {
        return true;
    }

    SphereGeometry();

    virtual ~SphereGeometry();

protected:
    bool create() override;

    void release() override;

private:
    void init(CallContext &ctx);

    bool assertData(core::moldyn::MultiParticleDataCall& call, CallContext& ctx);

    bool get_data_cb(core::Call& c);

    bool get_extents_cb(core::Call& c);

    core::CalleeSlot _out_geo_slot;

    core::CallerSlot _in_data_slot;

    core::CallerSlot _in_ctx_slot;

    MMOptixModule sphere_module_;

    std::vector<SBTRecord<device::SphereGeoData>> sbt_records_;

    CUdeviceptr _geo_buffer = 0;

    std::vector<CUdeviceptr> particle_data_;

    std::vector<CUdeviceptr> color_data_;

    OptixTraversableHandle _geo_handle;

    unsigned int _frame_id = std::numeric_limits<unsigned int>::max();

    std::size_t _data_hash = std::numeric_limits<std::size_t>::max();
};
} // namespace megamol::hpg::optix
