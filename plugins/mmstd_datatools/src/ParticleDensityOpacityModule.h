#ifndef AOWT_PARTICLE_DENSITY_OPACITY_MODULE_H_INCLUDED
#define AOWT_PARTICLE_DENSITY_OPACITY_MODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "vislib/RawStorage.h"
#include "vislib/memutils.h"
#include "TransferFunctionQuery.h"


namespace megamol {
namespace core {
namespace moldyn {
    class MultiParticleDataCall;
}
}

namespace stdplugin {
namespace datatools {


    class ParticleDensityOpacityModule : public core::Module {
    public:
        static const char *ClassName(void) {
            return "ParticleDensityOpacityModule";
        }
        static const char *Description(void) {
            return "Compute particle opacity based on local density.";
        }
        static bool IsAvailable(void) {
            return true;
        }
        ParticleDensityOpacityModule(void);
        virtual ~ParticleDensityOpacityModule(void);
    private:

        class Unlocker : public core::moldyn::MultiParticleDataCall::Unlocker {
        public:
            Unlocker(core::moldyn::MultiParticleDataCall::Unlocker *inner) :
                    core::moldyn::MultiParticleDataCall::Unlocker(), inner(inner) {
                // intentionally empty
            }
            virtual ~Unlocker(void) {
                this->Unlock();
            }
            virtual void Unlock(void) {
                if (this->inner != nullptr) {
                    this->inner->Unlock();
                    SAFE_DELETE(this->inner);
                }
            }
        private:
            core::moldyn::MultiParticleDataCall::Unlocker *inner;
        };

        enum class DensityAlgorithmType : int {
            grid = 1,
            listSepGrid = 2
        };

        enum class MapMode : int {
            AlphaOverwrite = 0,
            ColorRainbow = 1,
            ColorRainbowAlpha = 2,
            Luminance = 3,
            AlphaInvertOverwrite = 4,
        };

        virtual bool create(void);
        virtual void release(void);
        bool getDataCallback(core::Call& caller);
        bool getExtentCallback(core::Call& caller);

        void makeData(core::moldyn::MultiParticleDataCall *dat);

        size_t count_all_particles(core::moldyn::MultiParticleDataCall *dat);
        void compute_density_grid(core::moldyn::MultiParticleDataCall *dat, bool cycX, bool cycY, bool cycZ, float rad, float *f, int col_step, int col_off);
        void compute_density_grid_grouped(core::moldyn::MultiParticleDataCall *dat, bool cycX, bool cycY, bool cycZ, float rad, float *f, int col_step, int col_off);

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

}
}
}

#endif /* AOWT_PARTICLE_DENSITY_OPACITY_MODULE_H_INCLUDED */
