#ifndef AOWT_PARTICLE_DENSITY_OPACITY_MODULE_H_INCLUDED
#define AOWT_PARTICLE_DENSITY_OPACITY_MODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "Call.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"
#include "moldyn/MultiParticleDataCall.h"
#include "vislib/RawStorage.h"
#include "vislib/memutils.h"


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

        virtual bool create(void);
        virtual void release(void);
        bool getDataCallback(core::Call& caller);
        bool getExtentCallback(core::Call& caller);

        void makeData(core::moldyn::MultiParticleDataCall *dat);

        core::CalleeSlot putDataSlot;
        core::CallerSlot getDataSlot;
        core::CallerSlot getTFSlot;
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
        core::param::ParamSlot mapDensityToAlphaSlot;
        core::param::ParamSlot mapDensityToColorSlot;
        unsigned int lastFrame;
        size_t lastHash;
        vislib::RawStorage colData;

    };


}
}
}

#endif /* AOWT_PARTICLE_DENSITY_OPACITY_MODULE_H_INCLUDED */
