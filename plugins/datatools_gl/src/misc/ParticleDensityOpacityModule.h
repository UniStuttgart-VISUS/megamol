#ifndef AOWT_PARTICLE_DENSITY_OPACITY_MODULE_H_INCLUDED
#define AOWT_PARTICLE_DENSITY_OPACITY_MODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "datatools_gl/TransferFunctionQuery.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/RawStorage.h"
#include "vislib/memutils.h"


namespace megamol::moldyn {
class MultiParticleDataCall;
} // namespace megamol::moldyn

namespace megamol::datatools_gl::misc {


class ParticleDensityOpacityModule : public megamol::core::Module {
public:
    static const char* ClassName() {
        return "ParticleDensityOpacityModule";
    }
    static const char* Description() {
        return "Compute particle opacity based on local density.";
    }
    static bool IsAvailable() {
        return true;
    }
    ParticleDensityOpacityModule();
    ~ParticleDensityOpacityModule() override;

private:
    class Unlocker : public geocalls::MultiParticleDataCall::Unlocker {
    public:
        Unlocker(geocalls::MultiParticleDataCall::Unlocker* inner)
                : geocalls::MultiParticleDataCall::Unlocker()
                , inner(inner) {
            // intentionally empty
        }
        ~Unlocker() override {
            this->Unlock();
        }
        void Unlock() override {
            if (this->inner != nullptr) {
                this->inner->Unlock();
                SAFE_DELETE(this->inner);
            }
        }

    private:
        geocalls::MultiParticleDataCall::Unlocker* inner;
    };

    enum class DensityAlgorithmType : int { grid = 1, listSepGrid = 2 };

    enum class MapMode : int {
        AlphaOverwrite = 0,
        ColorRainbow = 1,
        ColorRainbowAlpha = 2,
        Luminance = 3,
        AlphaInvertOverwrite = 4,
    };

    bool create() override;
    void release() override;
    bool getDataCallback(core::Call& caller);
    bool getExtentCallback(core::Call& caller);

    void makeData(geocalls::MultiParticleDataCall* dat);

    size_t count_all_particles(geocalls::MultiParticleDataCall* dat);
    void compute_density_grid(geocalls::MultiParticleDataCall* dat, bool cycX, bool cycY, bool cycZ, float rad,
        float* f, int col_step, int col_off);
    void compute_density_grid_grouped(geocalls::MultiParticleDataCall* dat, bool cycX, bool cycY, bool cycZ, float rad,
        float* f, int col_step, int col_off);

    core::CalleeSlot putDataSlot;
    core::CallerSlot getDataSlot;
    core::param::ParamSlot rebuildButtonSlot;
    core::param::ParamSlot densityRadiusSlot;
    core::param::ParamSlot densityMinCountSlot;
    core::param::ParamSlot densityMaxCountSlot;
    core::param::ParamSlot densityComputeCountRangeSlot;
    core::param::ParamSlot opacityMinValSlot;
    core::param::ParamSlot opacityMaxValSlot;
    core::param::ParamSlot cyclBoundXSlot;
    core::param::ParamSlot cyclBoundYSlot;
    core::param::ParamSlot cyclBoundZSlot;
    core::param::ParamSlot mapModeSlot;
    unsigned int lastFrame;
    size_t lastHash;
    vislib::RawStorage colData;
    core::param::ParamSlot densitAlgorithmSlot;
    TransferFunctionQuery tfQuery;
    core::param::ParamSlot densityAutoComputeCountRangeSlot;
};

} // namespace megamol::datatools_gl::misc

#endif /* AOWT_PARTICLE_DENSITY_OPACITY_MODULE_H_INCLUDED */
