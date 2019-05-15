/*
 * AstroParticleConverter.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ASTROPARTICLECONVERTER_H_INCLUDED
#define MEGAMOLCORE_ASTROPARTICLECONVERTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ParamSlot.h"

#include "astro/AstroDataCall.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"

namespace megamol {
namespace astro {

class AstroParticleConverter : public core::Module {
public:
    static const char* ClassName(void) { return "AstroParticleConverter"; }
    static const char* Description(void) {
        return "Converts data contained in a AstroDataCall to a MultiParticleDataCall";
    }
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    AstroParticleConverter(void);

    /** Dtor. */
    virtual ~AstroParticleConverter(void);

protected:
    virtual bool create(void);
    virtual void release(void);

private:
    enum class ColoringMode : uint8_t {
        MASS = 0,
        INTERNAL_ENERGY = 1,
        SMOOTHING_LENGTH = 2,
        MOLECULAR_WEIGHT = 3,
        DENSITY = 4,
        GRAVITATIONAL_POTENTIAL = 5,
        IS_BARYON = 6,
        IS_STAR = 7,
        IS_WIND = 8,
        IS_STAR_FORMING_GAS = 9,
        IS_AGN = 10,
        IS_DARK_MATTER = 11
    };

    bool getData(core::Call& call);
    bool getExtent(core::Call& call);

    void calcMinMaxValues(const AstroDataCall& ast);
	void calcColorTable(const AstroDataCall& ast);

	glm::vec4 interpolateColor(const glm::vec4& minCol, const glm::vec4& midCol, const glm::vec4& maxCol, const float alpha, const bool useMidValue = false);

    core::param::ParamSlot colorModeSlot;
    core::param::ParamSlot minColorSlot;
    core::param::ParamSlot midColorSlot;
    core::param::ParamSlot maxColorSlot;
	core::param::ParamSlot useMidColorSlot;

    std::vector<glm::vec4> usedColors;

    core::CalleeSlot sphereDataSlot;
    core::CallerSlot astroDataSlot;
    size_t lastDataHash;
    size_t hashOffset;
	unsigned int lastFrame = 0;
    float valmin, valmax;
};

} // namespace astro
} // namespace megamol

#endif /* MEGAMOLCORE_ASTROPARTICLECONVERTER_H_INCLUDED */
