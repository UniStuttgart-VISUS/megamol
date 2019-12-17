/*
 * OSPRayCylinderGeometry.h
 *
 * Copyright (C) 2019 by MegaMol Team. Alle Rechte vorbehalten.
 */
// Make crappy clang-format f*** off:
// clang-format off

#pragma once

#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "OSPRay_plugin/AbstractOSPRayStructure.h"


namespace megamol {
namespace ospray {

    /** The MegaMol equivalent of our own arrow geometry in OSPRay. */
    class OSPRayCylinderGeometry : public core::Module {

    public:

        static inline constexpr const char *ClassName(void) {
           return "OSPRayCylinderGeometry";
        }

        static inline constexpr const char* Description(void) {
            return "Creates OSPRay cylinder geometry from patricle data.";
        }

        static inline constexpr bool IsAvailable(void) {
            return true;
        }

        OSPRayCylinderGeometry(void);

        virtual ~OSPRayCylinderGeometry(void);

    protected:

        typedef core::moldyn::SimpleSphericalParticles ParticleType;

        static bool checkState(core::param::ParamSlot& param, const bool reset);

        static std::tuple<std::size_t, std::size_t> getSizes(
            const ParticleType& particles);

        bool checkState(const bool reset);

        virtual bool create(void);

        virtual bool getData(core::Call &call);

        bool onGetData(megamol::core::Call& call);

        bool onGetDirty(megamol::core::Call& call);

        bool onGetExtents(megamol::core::Call& call);

        inline std::size_t getHash(void) const {
            auto retval = this->hashInput;
            retval ^= this->hashState + 0x9e3779b9 + (retval << 6)
                + (retval >> 2);
            return retval;
        }

        virtual void release(void);

    private:

        std::vector<std::uint8_t> data;
        unsigned int frameID;
        std::size_t hashInput;
        std::size_t hashState;
        core::param::ParamSlot paramRadius;
        core::param::ParamSlot paramScale;
        core::CallerSlot slotGetData;
        core::CalleeSlot slotInstantiate;

    };

} /* end namespace ospray */
} /* end namespace megamol */
