/*
 * AstroParticleConverter.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/ParamSlot.h"

#include "astro/AstroDataCall.h"
#include "geometry_calls/MultiParticleDataCall.h"

namespace megamol::astro {

class AstroParticleConverter : public core::Module {
public:
    static const char* ClassName() {
        return "AstroParticleConverter";
    }
    static const char* Description() {
        return "Converts data contained in a AstroDataCall to a MultiParticleDataCall";
    }
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    AstroParticleConverter();

    /** Dtor. */
    ~AstroParticleConverter() override;

protected:
    bool create() override;
    void release() override;

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
        IS_DARK_MATTER = 11,
        TEMPERATURE = 12,
        ENTROPY = 13,
        INTERNAL_ENERGY_DERIVATIVE = 14,
        SMOOTHING_LENGTH_DERIVATIVE = 15,
        MOLECULAR_WEIGHT_DERIVATIVE = 16,
        DENSITY_DERIVATIVE = 17,
        GRAVITATIONAL_POTENTIAL_DERIVATIVE = 18,
        TEMPERATURE_DERIVATIVE = 19,
        ENTROPY_DERIVATIVE = 20,
        AGN_DISTANCES = 21
    };

    bool getData(core::Call& call);
    bool getSpecialData(core::Call& call);
    bool getExtent(core::Call& call);

    void calcMinMaxValues(const AstroDataCall& ast);
    void calcColorTable(const AstroDataCall& ast);

    glm::vec4 interpolateColor(const glm::vec4& minCol, const glm::vec4& midCol, const glm::vec4& maxCol,
        const float alpha, const bool useMidValue = false);

    core::param::ParamSlot colorModeSlot;
    core::param::ParamSlot minColorSlot;
    core::param::ParamSlot midColorSlot;
    core::param::ParamSlot maxColorSlot;
    core::param::ParamSlot useMidColorSlot;

    core::param::ParamSlot minValueSlot;
    core::param::ParamSlot maxValueSlot;

    std::vector<glm::vec4> usedColors;

    core::CalleeSlot sphereDataSlot;
    core::CalleeSlot sphereSpecialSlot;
    core::CallerSlot astroDataSlot;
    size_t lastDataHash;
    size_t hashOffset;
    unsigned int lastFrame = 0;
    float valmin, valmax;
    float densityMin, densityMax;

    std::vector<glm::vec4> pos_;
    std::vector<glm::vec3> vel_;
    std::vector<float> dens_;
    std::vector<float> sl_;
    std::vector<float> temp_;
    std::vector<float> mass_;
    std::vector<float> mw_;
};

} // namespace megamol::astro
