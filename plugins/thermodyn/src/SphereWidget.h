#pragma once

#include <optional>

#include "mmcore/CallerSlot.h"
#include "mmcore/UniFlagCalls.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer3DModuleGL.h"

#include "implot.h"

namespace megamol::thermodyn {
class SphereWidget : public core::view::Renderer3DModuleGL {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "SphereWidget";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "SphereWidget";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    SphereWidget();

    virtual ~SphereWidget();

protected:
    bool create() override;

    void release() override;

private:
    bool Render(core::view::CallRender3DGL& call) override;

    bool GetExtents(core::view::CallRender3DGL& call) override;

    bool widget(float x, float y, std::size_t idx, core::moldyn::SimpleSphericalParticles const& parts,
        core::moldyn::SimpleSphericalParticles const* temps, core::moldyn::SimpleSphericalParticles const* dens,
        glm::mat4 vp, glm::ivec2 res);

    bool parse_data(core::moldyn::MultiParticleDataCall& in_parts, core::moldyn::MultiParticleDataCall* in_temps,
        core::moldyn::MultiParticleDataCall* in_dens, core::FlagCallRead_CPU& fcr,
        megamol::core::view::Camera_2 const& cam);

    bool OnMouseMove(double x, double y) override;

    core::CallerSlot in_data_slot_;

    core::CallerSlot in_temp_slot_;

    core::CallerSlot in_dens_slot_;

    core::CallerSlot flags_read_slot_;

    core::param::ParamSlot particlelist_slot_;

    ImPlotContext* ctx_ = nullptr;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    int frame_id_ = -1;

    float mouse_x_;
    float mouse_y_;
};
} // namespace megamol::thermodyn
