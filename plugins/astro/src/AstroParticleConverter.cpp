/*
 * AstroParticleConverter.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * All rights reserved.
 */

#include "stdafx.h"
#include "AstroParticleConverter.h"
#include <glm/gtc/type_ptr.hpp>

using namespace megamol;
using namespace megamol::astro;
using namespace megamol::core;
using namespace megamol::core::moldyn;

/*
 * AstroParticleConverter::AstroParticleConverter
 */
AstroParticleConverter::AstroParticleConverter(void)
    : Module()
    , sphereDataSlot("sphereData", "Output slot for the resulting sphere data")
    , astroDataSlot("astroData", "Input slot for astronomical data")
    , colorModeSlot("colorMode", "Coloring mode for the output particles")
    , minColorSlot("minColor", "minimum color of the used range")
    , midColorSlot("midColor", "median color of the used range")
    , maxColorSlot("maxColor", "maximum color of the used range")
    , useMidColorSlot("useMidColor", "Enables the usage of the mid color in the color interpolation")
    , lastDataHash(0)
    , hashOffset(0)
    , valmin(0.0f)
    , valmax(1.0f) {

    this->astroDataSlot.SetCompatibleCall<AstroDataCallDescription>();
    this->MakeSlotAvailable(&this->astroDataSlot);

    this->sphereDataSlot.SetCallback(
        MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(0), &AstroParticleConverter::getData);
    this->sphereDataSlot.SetCallback(
        MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(1), &AstroParticleConverter::getExtent);
    this->MakeSlotAvailable(&this->sphereDataSlot);

    param::EnumParam* enu = new param::EnumParam(static_cast<int>(ColoringMode::MASS));
    enu->SetTypePair(static_cast<int>(ColoringMode::MASS), "Mass");
    enu->SetTypePair(static_cast<int>(ColoringMode::INTERNAL_ENERGY), "Internal Energy");
    enu->SetTypePair(static_cast<int>(ColoringMode::SMOOTHING_LENGTH), "Smoothing Length");
    enu->SetTypePair(static_cast<int>(ColoringMode::MOLECULAR_WEIGHT), "Molecular Weight");
    enu->SetTypePair(static_cast<int>(ColoringMode::DENSITY), "Density");
    enu->SetTypePair(static_cast<int>(ColoringMode::GRAVITATIONAL_POTENTIAL), "Gravitational Potential");
    enu->SetTypePair(static_cast<int>(ColoringMode::IS_BARYON), "Baryon");
    enu->SetTypePair(static_cast<int>(ColoringMode::IS_STAR), "Star");
    enu->SetTypePair(static_cast<int>(ColoringMode::IS_WIND), "Wind");
    enu->SetTypePair(static_cast<int>(ColoringMode::IS_STAR_FORMING_GAS), "Star-forming Gas");
    enu->SetTypePair(static_cast<int>(ColoringMode::IS_AGN), "AGN");
    enu->SetTypePair(static_cast<int>(ColoringMode::IS_DARK_MATTER), "Dark Matter");
    enu->SetTypePair(static_cast<int>(ColoringMode::TEMPERATURE), "Temperature");
    enu->SetTypePair(static_cast<int>(ColoringMode::ENTROPY), "Entropy");
    this->colorModeSlot << enu;
    this->MakeSlotAvailable(&this->colorModeSlot);

    this->minColorSlot.SetParameter(new param::ColorParam("#146496"));
    this->MakeSlotAvailable(&this->minColorSlot);

    this->midColorSlot.SetParameter(new param::ColorParam("#f0f0f0"));
    this->MakeSlotAvailable(&this->midColorSlot);

    this->maxColorSlot.SetParameter(new param::ColorParam("#ae3b32"));
    this->MakeSlotAvailable(&this->maxColorSlot);

    this->useMidColorSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->useMidColorSlot);
}

/*
 * AstroParticleConverter::~AstroParticleConverter
 */
AstroParticleConverter::~AstroParticleConverter(void) { this->Release(); }

/*
 * AstroParticleConverter::create
 */
bool AstroParticleConverter::create(void) {
    // intentionally empty
    return true;
}

/*
 * AstroParticleConverter::release
 */
void AstroParticleConverter::release(void) {
    // intentionally empty
}

/*
 * AstroParticleConverter::getData
 */
bool AstroParticleConverter::getData(Call& call) {
    MultiParticleDataCall* mpdc = dynamic_cast<MultiParticleDataCall*>(&call);
    if (mpdc == nullptr) return false;

    AstroDataCall* ast = this->astroDataSlot.CallAs<AstroDataCall>();
    if (ast == nullptr) return false;

    ast->SetFrameID(mpdc->FrameID(), mpdc->IsFrameForced());
    //ast->SetUnlocker(nullptr, false);

    if ((*ast)(AstroDataCall::CallForGetData)) {
        bool freshMinMax = false;
        if (this->lastDataHash != ast->DataHash() || this->colorModeSlot.IsDirty() ||
            this->lastFrame != mpdc->FrameID()) {
            this->lastFrame = mpdc->FrameID();
            this->lastDataHash = ast->DataHash();
            this->colorModeSlot.ResetDirty();
            this->calcMinMaxValues(*ast);
            freshMinMax = true;
        }
        auto particleCount = ast->GetParticleCount();
        mpdc->SetDataHash(this->lastDataHash + this->hashOffset);
        mpdc->SetParticleListCount(1);
        MultiParticleDataCall::Particles& p = mpdc->AccessParticles(0);
        p.SetCount(particleCount);
        if (p.GetCount() > 0) {
            p.SetVertexData(MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ, ast->GetPositions()->data());
            if (freshMinMax || this->minColorSlot.IsDirty() || this->midColorSlot.IsDirty() ||
                this->maxColorSlot.IsDirty() || this->useMidColorSlot.IsDirty()) {
                this->calcColorTable(*ast);
                this->minColorSlot.ResetDirty();
                this->midColorSlot.ResetDirty();
                this->maxColorSlot.ResetDirty();
                this->useMidColorSlot.ResetDirty();
            }
            p.SetColourData(MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA, this->usedColors.data());
            p.SetColourMapIndexValues(this->valmin, this->valmax);
        }
        //ast->Unlock();
        return true;
    }
    ast->Unlock();
    return false;
}

/*
 * AstroParticleConverter::getExtent
 */
bool AstroParticleConverter::getExtent(Call& call) {
    MultiParticleDataCall* mpdc = dynamic_cast<MultiParticleDataCall*>(&call);
    if (mpdc == nullptr) return false;

    AstroDataCall* ast = this->astroDataSlot.CallAs<AstroDataCall>();
    if (ast == nullptr) return false;

    ast->SetUnlocker(nullptr, false);
    if ((*ast)(AstroDataCall::CallForGetExtent)) {
        mpdc->SetFrameCount(ast->FrameCount());
        mpdc->AccessBoundingBoxes() = ast->AccessBoundingBoxes();
        ast->Unlock();
        return true;
    }
    ast->Unlock();
    return false;
}

/*
 * AstroParticleConverter::calcMinMaxValues
 */
void AstroParticleConverter::calcMinMaxValues(const AstroDataCall& ast) {
    auto colmode = static_cast<ColoringMode>(this->colorModeSlot.Param<param::EnumParam>()->Value());
    switch (colmode) {
    case megamol::astro::AstroParticleConverter::ColoringMode::MASS:
        this->valmin = *std::min_element(ast.GetMass()->begin(), ast.GetMass()->end());
        this->valmax = *std::max_element(ast.GetMass()->begin(), ast.GetMass()->end());
        break;
    case megamol::astro::AstroParticleConverter::ColoringMode::INTERNAL_ENERGY:
        this->valmin = *std::min_element(ast.GetInternalEnergy()->begin(), ast.GetInternalEnergy()->end());
        this->valmax = *std::max_element(ast.GetInternalEnergy()->begin(), ast.GetInternalEnergy()->end());
        break;
    case megamol::astro::AstroParticleConverter::ColoringMode::SMOOTHING_LENGTH:
        this->valmin = *std::min_element(ast.GetSmoothingLength()->begin(), ast.GetSmoothingLength()->end());
        this->valmax = *std::max_element(ast.GetSmoothingLength()->begin(), ast.GetSmoothingLength()->end());
        break;
    case megamol::astro::AstroParticleConverter::ColoringMode::MOLECULAR_WEIGHT:
        this->valmin = *std::min_element(ast.GetMolecularWeights()->begin(), ast.GetMolecularWeights()->end());
        this->valmax = *std::max_element(ast.GetMolecularWeights()->begin(), ast.GetMolecularWeights()->end());
        break;
    case megamol::astro::AstroParticleConverter::ColoringMode::DENSITY:
        this->valmin = *std::min_element(ast.GetDensity()->begin(), ast.GetDensity()->end());
        this->valmax = *std::max_element(ast.GetDensity()->begin(), ast.GetDensity()->end());
        break;
    case megamol::astro::AstroParticleConverter::ColoringMode::GRAVITATIONAL_POTENTIAL:
        this->valmin =
            *std::min_element(ast.GetGravitationalPotential()->begin(), ast.GetGravitationalPotential()->end());
        this->valmax =
            *std::max_element(ast.GetGravitationalPotential()->begin(), ast.GetGravitationalPotential()->end());
        break;
    case megamol::astro::AstroParticleConverter::ColoringMode::TEMPERATURE:
        this->valmin = *std::min_element(ast.GetTemperature()->begin(), ast.GetTemperature()->end());
        this->valmax = *std::max_element(ast.GetTemperature()->begin(), ast.GetTemperature()->end());
        break;
    case megamol::astro::AstroParticleConverter::ColoringMode::ENTROPY:
        this->valmin = *std::min_element(ast.GetEntropy()->begin(), ast.GetEntropy()->end());
        this->valmax = *std::max_element(ast.GetEntropy()->begin(), ast.GetEntropy()->end());
        break;
    case megamol::astro::AstroParticleConverter::ColoringMode::IS_BARYON:
    case megamol::astro::AstroParticleConverter::ColoringMode::IS_STAR:
    case megamol::astro::AstroParticleConverter::ColoringMode::IS_WIND:
    case megamol::astro::AstroParticleConverter::ColoringMode::IS_STAR_FORMING_GAS:
    case megamol::astro::AstroParticleConverter::ColoringMode::IS_AGN:
    case megamol::astro::AstroParticleConverter::ColoringMode::IS_DARK_MATTER:
    default:
        this->valmin = 0.0f;
        this->valmax = 1.0f;
        break;
    }
}

/*
 * AstroParticleConverter::calcColorTable
 */
void AstroParticleConverter::calcColorTable(const AstroDataCall& ast) {
    float value = 0.0f;
    auto colmode = static_cast<ColoringMode>(this->colorModeSlot.Param<param::EnumParam>()->Value());
    this->usedColors.resize(ast.GetPositions()->size());
    auto minCol = glm::make_vec4(this->minColorSlot.Param<param::ColorParam>()->Value().data());
    auto midCol = glm::make_vec4(this->midColorSlot.Param<param::ColorParam>()->Value().data());
    auto maxCol = glm::make_vec4(this->maxColorSlot.Param<param::ColorParam>()->Value().data());
    auto useMid = this->useMidColorSlot.Param<param::BoolParam>()->Value();
    float denom = this->valmax - this->valmin;

    switch (colmode) {
    case megamol::astro::AstroParticleConverter::ColoringMode::MASS: {
        auto v = ast.GetMass();
        for (size_t i = 0; i < this->usedColors.size(); ++i) {
            float alpha = (v->at(i) - this->valmin) / denom;
            this->usedColors[i] = this->interpolateColor(minCol, midCol, maxCol, alpha, useMid);
        }
    } break;
    case megamol::astro::AstroParticleConverter::ColoringMode::INTERNAL_ENERGY: {
        auto v = ast.GetInternalEnergy();
        for (size_t i = 0; i < this->usedColors.size(); ++i) {
            float alpha = (v->at(i) - this->valmin) / denom;
            this->usedColors[i] = this->interpolateColor(minCol, midCol, maxCol, alpha, useMid);
        }
    } break;
    case megamol::astro::AstroParticleConverter::ColoringMode::SMOOTHING_LENGTH: {
        auto v = ast.GetSmoothingLength();
        for (size_t i = 0; i < this->usedColors.size(); ++i) {
            float alpha = (v->at(i) - this->valmin) / denom;
            this->usedColors[i] = this->interpolateColor(minCol, midCol, maxCol, alpha, useMid);
        }
    } break;
    case megamol::astro::AstroParticleConverter::ColoringMode::MOLECULAR_WEIGHT: {
        auto v = ast.GetMolecularWeights();
        for (size_t i = 0; i < this->usedColors.size(); ++i) {
            float alpha = (v->at(i) - this->valmin) / denom;
            this->usedColors[i] = this->interpolateColor(minCol, midCol, maxCol, alpha, useMid);
        }
    } break;
    case megamol::astro::AstroParticleConverter::ColoringMode::DENSITY: {
        auto v = ast.GetDensity();
        for (size_t i = 0; i < this->usedColors.size(); ++i) {
            float alpha = (v->at(i) - this->valmin) / denom;
            this->usedColors[i] = this->interpolateColor(minCol, midCol, maxCol, alpha, useMid);
        }
    } break;
    case megamol::astro::AstroParticleConverter::ColoringMode::GRAVITATIONAL_POTENTIAL: {
        auto v = ast.GetGravitationalPotential();
        for (size_t i = 0; i < this->usedColors.size(); ++i) {
            float alpha = (v->at(i) - this->valmin) / denom;
            this->usedColors[i] = this->interpolateColor(minCol, midCol, maxCol, alpha, useMid);
        }
    } break;
    case megamol::astro::AstroParticleConverter::ColoringMode::TEMPERATURE: {
        auto v = ast.GetTemperature();
        for (size_t i = 0; i < this->usedColors.size(); ++i) {
            float alpha = (v->at(i) - this->valmin) / denom;
            this->usedColors[i] = this->interpolateColor(minCol, midCol, maxCol, alpha, useMid);
        }
    } break;
    case megamol::astro::AstroParticleConverter::ColoringMode::ENTROPY: {
        auto v = ast.GetEntropy();
        for (size_t i = 0; i < this->usedColors.size(); ++i) {
            float alpha = (v->at(i) - this->valmin) / denom;
            this->usedColors[i] = this->interpolateColor(minCol, midCol, maxCol, alpha, useMid);
        }
    } break;
    case megamol::astro::AstroParticleConverter::ColoringMode::IS_BARYON: {
        auto v = ast.GetIsBaryonFlags();
        for (size_t i = 0; i < this->usedColors.size(); ++i) {
            v->at(i) ? this->usedColors[i] = maxCol : this->usedColors[i] = minCol;
        }
    } break;
    case megamol::astro::AstroParticleConverter::ColoringMode::IS_STAR: {
        auto v = ast.GetIsStarFlags();
        for (size_t i = 0; i < this->usedColors.size(); ++i) {
            v->at(i) ? this->usedColors[i] = maxCol : this->usedColors[i] = minCol;
        }
    } break;
    case megamol::astro::AstroParticleConverter::ColoringMode::IS_WIND: {
        auto v = ast.GetIsWindFlags();
        for (size_t i = 0; i < this->usedColors.size(); ++i) {
            v->at(i) ? this->usedColors[i] = maxCol : this->usedColors[i] = minCol;
        }
    } break;
    case megamol::astro::AstroParticleConverter::ColoringMode::IS_STAR_FORMING_GAS: {
        auto v = ast.GetIsStarFormingGasFlags();
        for (size_t i = 0; i < this->usedColors.size(); ++i) {
            v->at(i) ? this->usedColors[i] = maxCol : this->usedColors[i] = minCol;
        }
    } break;
    case megamol::astro::AstroParticleConverter::ColoringMode::IS_AGN: {
        auto v = ast.GetIsAGNFlags();
        for (size_t i = 0; i < this->usedColors.size(); ++i) {
            v->at(i) ? this->usedColors[i] = maxCol : this->usedColors[i] = minCol;
        }
    } break;
    case megamol::astro::AstroParticleConverter::ColoringMode::IS_DARK_MATTER: { // inverse case to IS_BARYON
        auto v = ast.GetIsBaryonFlags();
        for (size_t i = 0; i < this->usedColors.size(); ++i) {
            v->at(i) ? this->usedColors[i] = minCol : this->usedColors[i] = maxCol;
        }
    } break;
    default:
        break;
    }
}

/*
 * AstroParticleConverter::interpolateColor
 */
glm::vec4 AstroParticleConverter::interpolateColor(const glm::vec4& minCol, const glm::vec4& midCol,
    const glm::vec4& maxCol, const float alpha, const bool useMidValue) {
    if (!useMidValue) return glm::mix(minCol, maxCol, alpha);
    if (alpha < 0.5f) return glm::mix(minCol, midCol, alpha * 2.0f);
    return glm::mix(midCol, maxCol, (alpha - 0.5f) * 2.0f);
}
