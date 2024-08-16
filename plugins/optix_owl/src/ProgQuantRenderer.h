#pragma once

#include <vector>

#include "mmcore/param/ParamSlot.h"

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

#include <owl/common/math/vec.h>
#include <owl/owl.h>

#include "framestate.h"
#include "particle.h"
#include "progquant.h"

#include "BaseRenderer.h"

namespace megamol::optix_owl {
class ProgQuantRenderer : public BaseRenderer {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ProgQuantRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    ProgQuantRenderer();

    virtual ~ProgQuantRenderer();

protected:
    bool create() override;

    void release() override;

private:
    bool assertData(geocalls::MultiParticleDataCall const& call) override;

    bool data_param_is_dirty() override;
    void data_param_reset_dirty() override;

    core::param::ParamSlot threshold_slot_;

    OWLModule pkd_module_;
    OWLGeom geom_;

    OWLBuffer treeletBuffer_ = 0;

    std::vector<device::Particle> particles_;
    std::vector<device::ProgQuantParticle> comp_particles_;

    unsigned int frame_id_ = 0;
    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();
};
} // namespace megamol::optix_owl
