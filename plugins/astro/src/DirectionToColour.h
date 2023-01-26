/*
 * DirectionToColour.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <array>

#include <glm/glm.hpp>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

#include "geometry_calls/MultiParticleDataCall.h"


namespace megamol {
namespace astro {

/// <summary>
/// Converts from <see cref="AstroDataCall" /> to a table for data
/// visualisation.
/// </summary>
class DirectionToColour : public core::Module {

public:
    /// <summary>
    /// Specifies the possible colouring modes.
    /// </summary>
    enum Mode {
        Hsl = 0,
        HelmholtzComplementary,
        IttenComplementary,
        MaxHelmholtzComplementary,
        MaxIttenComplementary,
        SaturationLightness,
        HuesLightness,
    };

    static inline const char* ClassName() {
        return "DirectionToColour";
    }

    static inline const char* Description() {
        return "Generates particle colours based on directional vectors "
               "and the HSL colour model.";
    }

    static bool IsAvailable() {
        return true;
    }

    /// <summary>
    /// Initialises a new instance.
    /// </summary>
    DirectionToColour();

    /// <summary>
    /// Finalises the instance.
    /// </summary>
    ~DirectionToColour() override;

protected:
    bool create() override;

    void release() override;

private:
    static float angle(const glm::vec2& v1, const glm::vec2& v2);

    static const std::uint8_t* getDirections(const geocalls::SimpleSphericalParticles& particles);

    static void hsl2Rgb(float* dst, const float h, const float s, const float l);

    static std::vector<float> makeComplementaryColouring(const std::uint8_t* directions,
        const std::uint64_t cntParticles, const std::size_t stride, const glm::vec3& x1, const glm::vec3& x2,
        const glm::vec3& y1, const glm::vec3& y2, const glm::vec3& z1, const glm::vec3& z2, const bool mix);

    static std::vector<float> makeHslColouring(const std::uint8_t* directions, const std::uint64_t cntParticles,
        const std::size_t stride, const Mode mode, const std::array<float, 4>& baseColour1,
        const std::array<float, 4>& baseColour2);

    static inline float min3(const float x, const float y, const float z) {
        return (std::min)((std::min)(x, y), z);
    }

    static std::array<float, 3> rgb2Hsl(const float r, const float g, const float b);

    static inline std::array<float, 3> rgb2Hsl(const std::array<float, 4>& rgb) {
        return rgb2Hsl(rgb[0], rgb[1], rgb[2]);
    }

    bool getData(core::Call& call);

    bool getExtent(core::Call& call);

    inline std::size_t getHash() {
        auto retval = this->hashData;
        retval ^= this->hashState + 0x9e3779b9 + (retval << 6) + (retval >> 2);
        return retval;
    }

    std::vector<std::vector<float>> colours;
    unsigned int frameID;
    std::size_t hashData;
    std::size_t hashState;
    core::param::ParamSlot paramColour1;
    core::param::ParamSlot paramColour2;
    core::param::ParamSlot paramMode;
    core::CallerSlot slotInput;
    core::CalleeSlot slotOutput;
};

} /* end namespace astro */
} /* end namespace megamol */
