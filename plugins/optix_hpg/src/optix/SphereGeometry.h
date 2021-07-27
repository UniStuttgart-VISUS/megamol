#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "optix/CallContext.h"

#include "cuda.h"

#include "SBT.h"

#include "sphere.h"

#include "MMOptixModule.h"

#include "optix/Context.h"

namespace megamol::optix_hpg {
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
    void init(Context const& ctx);

    bool assertData(core::moldyn::MultiParticleDataCall& call, Context const& ctx);

    bool get_data_cb(core::Call& c);

    bool get_extents_cb(core::Call& c);

    core::CalleeSlot _out_geo_slot;

    core::CallerSlot _in_data_slot;

    core::CallerSlot flags_read_slot_;

    core::CallerSlot flags_write_slot_;

    MMOptixModule sphere_module_;

    MMOptixModule sphere_occlusion_module_;

    MMOptixModule sphere_picking_module_;

    std::vector<SBTRecord<device::SphereGeoData>> sbt_records_;

    std::array<OptixProgramGroup, 3> program_groups_;

    CUdeviceptr _geo_buffer = 0;

    std::vector<CUdeviceptr> particle_data_;

    std::vector<CUdeviceptr> color_data_;

    OptixTraversableHandle _geo_handle;

    unsigned int _frame_id = std::numeric_limits<unsigned int>::max();

    std::size_t _data_hash = std::numeric_limits<std::size_t>::max();

    std::vector<uint64_t> prefix_count_;

    uint32_t flags_version_ = 0;
};
} // namespace megamol::optix_hpg
