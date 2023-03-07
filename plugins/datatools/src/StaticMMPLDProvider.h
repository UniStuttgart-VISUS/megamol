#pragma once

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "mmpld.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/mathtypes.h"

namespace megamol::datatools {

class StaticMMPLDProvider : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "StaticMMPLDProvider";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Reads a set of static MMPLDs";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    StaticMMPLDProvider();

    /** Dtor. */
    ~StaticMMPLDProvider() override;

protected:
    bool create() override;

    void release() override;

    core::param::ParamSlot filenamesSlot;

private:
    static geocalls::SimpleSphericalParticles::ColourDataType ColorTypeTranslator(mmpld::color_type ct) {
        switch (ct) {
        case mmpld::color_type::UINT8_RGB:
            return geocalls::SimpleSphericalParticles::COLDATA_UINT8_RGB;
        case mmpld::color_type::UINT8_RGBA:
            return geocalls::SimpleSphericalParticles::COLDATA_UINT8_RGBA;
        case mmpld::color_type::FLOAT_RGB:
            return geocalls::SimpleSphericalParticles::COLDATA_FLOAT_RGB;
        case mmpld::color_type::FLOAT_RGBA:
            return geocalls::SimpleSphericalParticles::COLDATA_FLOAT_RGBA;
        case mmpld::color_type::FLOAT_I:
            return geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I;
        case mmpld::color_type::DOUBLE_I:
            return geocalls::SimpleSphericalParticles::COLDATA_DOUBLE_I;
        case mmpld::color_type::SHORT_RGBA:
            return geocalls::SimpleSphericalParticles::COLDATA_USHORT_RGBA;
        }

        return geocalls::SimpleSphericalParticles::COLDATA_NONE;
    }

    bool assertData(geocalls::MultiParticleDataCall& outCall);

    bool getDataCallback(core::Call& c);

    bool getExtentCallback(core::Call& c);

    core::CalleeSlot outDataSlot;

    std::vector<mmpld::frame_t> output_frames;

    vislib::math::Cuboid<float> gbbox;

    size_t hash = 0;
}; // end class StaticMMPLDProvider

} // namespace megamol::datatools
