/*
 * DirectionToColour.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * All rights reserved.
 */

#include "DirectionToColour.h"

#include <cassert>
#include <limits>

#include <glm/gtc/constants.hpp>

#include <glm/gtx/vector_angle.hpp>

#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"

#include "mmcore/utility/log/Log.h"


/*
 * megamol::astro::DirectionToColour::DirectionToColour
 */
megamol::astro::DirectionToColour::DirectionToColour()
        : Module()
        , frameID(0)
        , hashData((std::numeric_limits<std::size_t>::max)())
        , hashState((std::numeric_limits<std::size_t>::max)())
        , paramColour1("colour1", "Sets the base colour for saturation-based colouring.")
        , paramColour2("colour2", "Sets the second colour for two-colour mode.")
        , paramMode("mode", "Changes the colouring mode for mapping direction to colour.")
        , slotInput("input", "Obtains the input particle data.")
        , slotOutput("output", "Output of the coloured data.") {
    using namespace geocalls;

    /* Configure and publish the slots. */
    this->slotInput.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->slotInput);

    this->slotOutput.SetCallback(
        MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(0), &DirectionToColour::getData);
    this->slotOutput.SetCallback(
        MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(1), &DirectionToColour::getExtent);
    this->MakeSlotAvailable(&this->slotOutput);

    /* Configure and publish parameters. */
    {
        auto param = new core::param::EnumParam(Mode::Hsl);
        param->SetTypePair(Mode::Hsl, "HSL plane & lightness");
        param->SetTypePair(Mode::HelmholtzComplementary, "Mixed complementary colours (Helmholtz)");
        param->SetTypePair(Mode::IttenComplementary, "Mixed complementary colours (Itten)");
        param->SetTypePair(Mode::MaxHelmholtzComplementary, "Maximum complementary colours (Helmholtz)");
        param->SetTypePair(Mode::MaxIttenComplementary, "Maximum complementary colours (Itten)");
        param->SetTypePair(Mode::SaturationLightness, "Saturation plane & lightness");
        param->SetTypePair(Mode::HuesLightness, "Hue interpolation & lightness");
        this->paramMode << param;
        this->MakeSlotAvailable(&this->paramMode);
    }

    this->paramColour1 << new core::param::ColorParam("Chartreuse");
    this->MakeSlotAvailable(&this->paramColour1);

    this->paramColour2 << new core::param::ColorParam("Cyan");
    this->MakeSlotAvailable(&this->paramColour2);
}


/*
 * megamol::astro::DirectionToColour::~DirectionToColour
 */
megamol::astro::DirectionToColour::~DirectionToColour() {
    this->Release();
}


/*
 * megamol::astro::DirectionToColour::create
 */
bool megamol::astro::DirectionToColour::create() {
    return true;
}


/*
 * megamol::astro::DirectionToColour::release
 */
void megamol::astro::DirectionToColour::release() {}


/*
 * megamol::astro::DirectionToColour::angle
 */
float megamol::astro::DirectionToColour::angle(const glm::vec2& v1, const glm::vec2& v2) {
    // Non-shitty version of angle between two vectors from
    // https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors
    auto dot = glm::dot(v1, v2);
    auto det = v1.x * v2.y - v1.y * v2.x; // cross for vec2 seems to be missing.
    return std::atan2(det, dot);
}


/*
 * megamol::astro::DirectionToColour::getDirections
 */
const std::uint8_t* megamol::astro::DirectionToColour::getDirections(
    const geocalls::SimpleSphericalParticles& particles) {
    using geocalls::SimpleSphericalParticles;
    using megamol::core::utility::log::Log;

    switch (particles.GetDirDataType()) {
    case SimpleSphericalParticles::DIRDATA_FLOAT_XYZ:
        return static_cast<const std::uint8_t*>(particles.GetDirData());

    default:
        Log::DefaultLog.WriteWarn("The input particles do not have "
                                  "directional information.");
        return nullptr;
    }
}


/*
 * megamol::astro::DirectionToColour::hsl2Rgb
 */
void megamol::astro::DirectionToColour::hsl2Rgb(float* dst, const float h, const float s, const float l) {
    assert(dst != nullptr);
    auto a = s * (std::min)(l, 1.0f - l);
    auto f = [a, h, l](const int n) {
        auto k = static_cast<int>(n + h / 30.0f) % 12;
        return l - a * (std::max)(min3(k - 3.0f, 9.0f - k, 1.0f), -1.0f);
    };

    dst[0] = f(0);
    dst[1] = f(8);
    dst[2] = f(4);
}


/*
 * megamol::astro::DirectionToColour::makeComplementaryColouring
 */
std::vector<float> megamol::astro::DirectionToColour::makeComplementaryColouring(const std::uint8_t* directions,
    const std::uint64_t cntParticles, const std::size_t stride, const glm::vec3& x1, const glm::vec3& x2,
    const glm::vec3& y1, const glm::vec3& y2, const glm::vec3& z1, const glm::vec3& z2, const bool mix) {
    assert(directions != nullptr);

    std::vector<float> retval(3 * cntParticles);

    for (UINT64 i = 0; i < cntParticles; ++i, directions += stride) {
        auto ptr = reinterpret_cast<const float*>(directions);
        glm::vec3 dir(ptr[0], ptr[1], ptr[2]);
        auto len = glm::length(dir);

        if (len != 0.0f) {
            /* Vector has a direction, so compute a colour. */
            dir /= len;
            dir *= 0.5f;
            dir += 0.5f;

            glm::vec3 hsl;

            if (!mix && (dir.x > dir.y) && (dir.x > dir.z)) {
                hsl = glm::mix(x1, x2, dir.x);

            } else if (!mix && (dir.y > dir.x) && (dir.y > dir.z)) {
                hsl = glm::mix(y1, y2, dir.y);

            } else if (!mix && (dir.z > dir.x) && (dir.z > dir.y)) {
                hsl = glm::mix(z1, z2, dir.z);

            } else {
                auto x = glm::mix(x1, x2, dir.x);
                auto y = glm::mix(y1, y2, dir.y);
                auto z = glm::mix(z1, z2, dir.z);
                hsl = dir.x * +dir.y * y + dir.z * z;
            }

            DirectionToColour::hsl2Rgb(retval.data() + 3 * i, hsl.x, hsl.y, hsl.z);

        } else {
            /* Zero-length vectors are black. */
            retval[3 * i + 0] = retval[3 * i + 1] = retval[3 * i + 2] = 0.0f;
        }
    }

    return retval;
}


/*
 * megamol::astro::DirectionToColour::makeHslColouring
 */
std::vector<float> megamol::astro::DirectionToColour::makeHslColouring(const std::uint8_t* directions,
    const std::uint64_t cntParticles, const std::size_t stride, const Mode mode,
    const std::array<float, 4>& baseColour1, const std::array<float, 4>& baseColour2) {
    assert(directions != nullptr);
    static constexpr const auto PI = glm::pi<float>();
    static constexpr const auto SCALE = 0.8f;
    static const auto BASE_HUE1 = rgb2Hsl(baseColour1).front();
    static const auto BASE_HUE2 = rgb2Hsl(baseColour2).front();
    static const glm::vec2 X(1.0f, 0.0f);

    std::vector<float> retval(3 * cntParticles);

    for (UINT64 i = 0; i < cntParticles; ++i, directions += stride) {
        auto ptr = reinterpret_cast<const float*>(directions);
        glm::vec3 dir(ptr[0], ptr[1], ptr[2]);
        auto len = glm::length(dir);

        if (len != 0.0f) {
            /* Vector has a direction, so compute a colour. */
            dir /= len;

            // Determine the lightness, which is the length on the y-axis. Note
            // that lightness must be within [0, 1] for colour conversion, so we
            // need to rescale it. In order from vectors being black, we
            // truncate the range to SCALE percent of what is possible.
            auto lightness = dir.y;
            lightness *= 0.5f * SCALE;
            lightness += 0.5f + 0.5f * (1.0f - SCALE);

            // Project direction on xz-plane, which defines the hue.
            auto proj = glm::vec2(dir.x, dir.y);
            proj = glm::normalize(proj);

            // Determine the angle with the x-axis, which we use for hue.
            auto hue = DirectionToColour::angle(X, proj);
            if (hue < 0.0f) {
                hue = 2.0f * PI - hue;
            }
            assert(hue >= 0.0f);

            switch (mode) {
            case Mode::SaturationLightness:
                hue /= (2.0f * PI);
                assert(hue <= 2.0f * PI);
                DirectionToColour::hsl2Rgb(retval.data() + 3 * i, BASE_HUE1, hue, lightness);
                break;

            case Mode::HuesLightness:
                hue /= (2.0f * PI);
                hue = glm::mix(BASE_HUE1, BASE_HUE2, hue);
                DirectionToColour::hsl2Rgb(retval.data() + 3 * i, hue, 1.0f, lightness);
                break;

            default:
                hue = glm::degrees(hue);
                DirectionToColour::hsl2Rgb(retval.data() + 3 * i, hue, 1.0f, lightness);
                break;
            }

        } else {
            /* Zero-length vectors are black. */
            retval[3 * i + 0] = retval[3 * i + 1] = retval[3 * i + 2] = 0.0f;
        }
    }

    return retval;
}


/*
 * megamol::astro::DirectionToColour::rgb2Hsl
 */
std::array<float, 3> megamol::astro::DirectionToColour::rgb2Hsl(const float r, const float g, const float b) {
    const auto minRgb = (std::min)((std::min)(r, g), b);
    const auto maxRgb = (std::max)((std::max)(r, g), b);
    const auto delta = maxRgb - minRgb;
    std::array<float, 3> retval;

    retval[2] = (maxRgb + minRgb) / 2.0f;

    if (delta == 0.0f) {
        retval[0] = retval[1] = 0.0f;

    } else {
        retval[1] = (retval[2] <= 0.5) ? (delta / (maxRgb + minRgb)) : (delta / (2 - maxRgb - minRgb));

        float hue;

        if (r == maxRgb) {
            hue = ((g - b) / 6.0f) / delta;

        } else if (g == maxRgb) {
            hue = (1.0f / 3.0f) + ((b - r) / 6.0f) / delta;

        } else {
            hue = (2.0f / 3.0f) + ((r - g) / 6.0f) / delta;
        }

        if (hue < 0.0f)
            hue += 1.0f;
        if (hue > 1.0f)
            hue -= 1.0f;

        retval[0] = hue * 360.0f;
    }

    return retval;
}


/*
 * megamol::astro::DirectionToColour::getData
 */
bool megamol::astro::DirectionToColour::getData(core::Call& call) {
    using core::param::ColorParam;
    using core::param::EnumParam;
    using geocalls::MultiParticleDataCall;
    using geocalls::SimpleSphericalParticles;
    using megamol::core::utility::log::Log;

    auto src = this->slotInput.CallAs<MultiParticleDataCall>();
    auto dst = dynamic_cast<MultiParticleDataCall*>(&call);
    const auto isLocalChange =
        this->paramColour1.IsDirty() || this->paramColour2.IsDirty() || this->paramMode.IsDirty();

    /* Sanity checks. */
    if (src == nullptr) {
        Log::DefaultLog.WriteError("The input slot of %hs is invalid.", DirectionToColour::ClassName());
        return false;
    }

    if (dst == nullptr) {
        Log::DefaultLog.WriteError("The output slot of %hs is invalid.", DirectionToColour::ClassName());
        return false;
    }

    /* Request the source data. */
    *src = *dst;
    if (!(*src)(0)) {
        Log::DefaultLog.WriteError("The call to %hs failed in %hs.", MultiParticleDataCall::FunctionName(0),
            MultiParticleDataCall::ClassName());
        return false;
    }
    *dst = *src;

    /* Generate the colours. */
    if (isLocalChange || (this->hashData != src->DataHash()) || (this->frameID != src->FrameID())) {
        const auto mode = static_cast<Mode>(this->paramMode.Param<EnumParam>()->Value());
        this->colours.clear();

        for (UINT i = 0; i < dst->GetParticleListCount(); ++i) {
            auto& particles = dst->AccessParticles(i);
            auto directions = getDirections(particles);

            if (directions != nullptr) {
                switch (mode) {
                case Mode::HelmholtzComplementary:
                case Mode::MaxHelmholtzComplementary:
                    this->colours.emplace_back(DirectionToColour::makeComplementaryColouring(directions,
                        particles.GetCount(), particles.GetDirDataStride(), glm::vec3(0.0f, 1.0f, 0.5f), // red
                        glm::vec3(180.0f, 1.0f, 0.5f),                                                   // cyan
                        glm::vec3(120.0f, 1.0f, 0.5f),                                                   // green
                        glm::vec3(300.0f, 1.0f, 0.5f),                                                   // magenta
                        glm::vec3(240.0f, 1.0f, 0.5f),                                                   // blue
                        glm::vec3(60.0f, 1.0f, 0.5f),                                                    // yellow
                        mode == Mode::HelmholtzComplementary));
                    break;

                case Mode::IttenComplementary:
                case Mode::MaxIttenComplementary:
                    this->colours.emplace_back(DirectionToColour::makeComplementaryColouring(directions,
                        particles.GetCount(), particles.GetDirDataStride(), glm::vec3(0.0f, 1.0f, 0.5f), // red
                        glm::vec3(120.0f, 1.0f, 0.5f),                                                   // green
                        glm::vec3(240, 1.0f, 0.5f),                                                      // blue
                        glm::vec3(30.0f, 1.0f, 0.5f),                                                    // orange
                        glm::vec3(60.0f, 1.0f, 0.5f),                                                    // yellow
                        glm::vec3(270.0f, 1.0f, 0.5f),                                                   // violet
                        mode == Mode::IttenComplementary));
                    break;

                case Mode::SaturationLightness:
                    this->colours.emplace_back(DirectionToColour::makeHslColouring(directions, particles.GetCount(),
                        particles.GetDirDataStride(), mode, this->paramColour1.Param<ColorParam>()->Value(),
                        this->paramColour2.Param<ColorParam>()->Value()));
                    break;

                case Mode::Hsl:
                default:
                    this->colours.emplace_back(DirectionToColour::makeHslColouring(directions, particles.GetCount(),
                        particles.GetDirDataStride(), mode, this->paramColour1.Param<ColorParam>()->Value(),
                        this->paramColour2.Param<ColorParam>()->Value()));
                    break;
                }

            } else {
                this->colours.emplace_back();
            }
        }

        this->frameID = src->FrameID();
        this->hashData = src->DataHash();

        if (isLocalChange) {
            this->paramColour1.ResetDirty();
            this->paramColour2.ResetDirty();
            this->paramMode.ResetDirty();
            ++this->hashState;
        }
    }

    /* Assign the new colours. */
    assert(this->colours.size() == dst->GetParticleListCount());
    for (UINT i = 0; i < dst->GetParticleListCount(); ++i) {
        auto& particles = dst->AccessParticles(i);

        if (!this->colours[i].empty()) {
            particles.SetColourData(
                SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_RGB, this->colours[i].data());
        }
    }

    dst->SetDataHash(this->getHash());

    return true;
}


/*
 * megamol::astro::DirectionToColour::getExtent
 */
bool megamol::astro::DirectionToColour::getExtent(core::Call& call) {
    using geocalls::MultiParticleDataCall;
    using megamol::core::utility::log::Log;

    auto src = this->slotInput.CallAs<MultiParticleDataCall>();
    auto dst = dynamic_cast<MultiParticleDataCall*>(&call);

    /* Sanity checks. */
    if (src == nullptr) {
        Log::DefaultLog.WriteError("The input slot of %hs is invalid.", DirectionToColour::ClassName());
        return false;
    }

    if (dst == nullptr) {
        Log::DefaultLog.WriteError("The output slot of %hs is invalid.", DirectionToColour::ClassName());
        return false;
    }

    *src = *dst;
    auto retval = (*src)(1);

    if (retval) {
        *dst = *src;
        dst->SetDataHash(this->getHash());
    }

    return retval;
}
