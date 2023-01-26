/*
 * SimpleAstroFilter.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "SimpleAstroFilter.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FloatParam.h"
#include <climits>
#include <numeric>

using namespace megamol;
using namespace megamol::astro;
using namespace megamol::core;

/*
 * SimpleAstroFilter::SimpleAstroFilter
 */
SimpleAstroFilter::SimpleAstroFilter()
        : Module()
        , particlesOutSlot("particlesOut", "Output slot for the filtered astro particle data")
        , particlesInSlot("particlesIn", "Input slot for the astro particle data")
        , showOnlyBaryonParam("showOnlyBaryons", "")
        , showOnlyDarkMatterParam("showOnlyDarkMatter", "")
        , showOnlyStarsParam("showOnlyStars", "")
        , showOnlyWindParam("showOnlyWind", "")
        , showOnlyStarFormingGasParam("showOnlyStarFormingGas", "")
        , showOnlyAGNsParam("showOnlyAGNs", "")
        , minVelocityMagnitudeParam("velocityMagnitude::min", "")
        , maxVelocityMagnitudeParam("velocityMagnitude::max", "")
        , filterVelocityMagnitudeParam("velocityMagnitude::filter", "")
        , minTemperatureParam("temperature::min", "")
        , maxTemperatureParam("temperature::max", "")
        , filterTemperatureParam("temperature::filter", "")
        , minMassParam("mass::min", "")
        , maxMassParam("mass::max", "")
        , filterMassParam("mass::filter", "")
        , minInternalEnergyParam("internalEnergy::min", "")
        , maxInternalEnergyParam("internalEnergy::max", "")
        , filterInternalEnergyParam("internalEnergy::filter", "")
        , minSmoothingLengthParam("smoothingLength::min", "")
        , maxSmoothingLengthParam("smoothingLength::max", "")
        , filterSmoothingLengthParam("smoothingLength::filter", "")
        , minMolecularWeightParam("molecularWeight::min", "")
        , maxMolecularWeightParam("molecularWeight::max", "")
        , filterMolecularWeightParam("molecularWeight::filter", "")
        , minDensityParam("density::min", "")
        , maxDensityParam("density::max", "")
        , filterDensityParam("density::filter", "")
        , minGravitationalPotentialParam("gravitationalPotential::min", "")
        , maxGravitationalPotentialParam("gravitationalPotential::max", "")
        , filterGravitationalPotentialParam("gravitationalPotential::filter", "")
        , minEntropyParam("entropy::min", "")
        , maxEntropyParam("entropy::max", "")
        , filterEntropyParam("entropy::filter", "")
        , minAgnDistanceParam("agndistance::min", "")
        , maxAgnDistanceParam("agndistance::max", "")
        , filterAgnDistanceParam("agndistance::filter", "")
        , fillFilterButtonParam("fillValues", "")
        , hashOffset(0)
        , refilter(true)
        , lastDataHash(0)
        , lastTimestep(0) {

    this->particlesInSlot.SetCompatibleCall<AstroDataCallDescription>();
    this->MakeSlotAvailable(&this->particlesInSlot);

    this->particlesOutSlot.SetCallback(AstroDataCall::ClassName(),
        AstroDataCall::FunctionName(AstroDataCall::CallForGetData), &SimpleAstroFilter::getData);
    this->particlesOutSlot.SetCallback(AstroDataCall::ClassName(),
        AstroDataCall::FunctionName(AstroDataCall::CallForGetExtent), &SimpleAstroFilter::getExtent);
    this->MakeSlotAvailable(&this->particlesOutSlot);

    this->showOnlyBaryonParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showOnlyBaryonParam);

    this->showOnlyDarkMatterParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showOnlyDarkMatterParam);

    this->showOnlyStarsParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showOnlyStarsParam);

    this->showOnlyWindParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showOnlyWindParam);

    this->showOnlyStarFormingGasParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showOnlyStarFormingGasParam);

    this->showOnlyAGNsParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showOnlyAGNsParam);

    // numeric filters
    this->minVelocityMagnitudeParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->minVelocityMagnitudeParam);
    this->maxVelocityMagnitudeParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->maxVelocityMagnitudeParam);
    this->filterVelocityMagnitudeParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->filterVelocityMagnitudeParam);

    this->minTemperatureParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->minTemperatureParam);
    this->maxTemperatureParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->maxTemperatureParam);
    this->filterTemperatureParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->filterTemperatureParam);

    this->minMassParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->minMassParam);
    this->maxMassParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->maxMassParam);
    this->filterMassParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->filterMassParam);

    this->minInternalEnergyParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->minInternalEnergyParam);
    this->maxInternalEnergyParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->maxInternalEnergyParam);
    this->filterInternalEnergyParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->filterInternalEnergyParam);

    this->minSmoothingLengthParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->minSmoothingLengthParam);
    this->maxSmoothingLengthParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->maxSmoothingLengthParam);
    this->filterSmoothingLengthParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->filterSmoothingLengthParam);

    this->minMolecularWeightParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->minMolecularWeightParam);
    this->maxMolecularWeightParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->maxMolecularWeightParam);
    this->filterMolecularWeightParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->filterMolecularWeightParam);

    this->minDensityParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->minDensityParam);
    this->maxDensityParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->maxDensityParam);
    this->filterDensityParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->filterDensityParam);

    this->minGravitationalPotentialParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->minGravitationalPotentialParam);
    this->maxGravitationalPotentialParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->maxGravitationalPotentialParam);
    this->filterGravitationalPotentialParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->filterGravitationalPotentialParam);

    this->minEntropyParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->minEntropyParam);
    this->maxEntropyParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->maxEntropyParam);
    this->filterEntropyParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->filterEntropyParam);

    this->minAgnDistanceParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->minAgnDistanceParam);
    this->maxAgnDistanceParam.SetParameter(new param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->maxAgnDistanceParam);
    this->filterAgnDistanceParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->filterAgnDistanceParam);

    this->fillFilterButtonParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_F));
    this->MakeSlotAvailable(&this->fillFilterButtonParam);

    this->initFields();
}

/*
 * SimpleAstroFilter::~SimpleAstroFilter
 */
SimpleAstroFilter::~SimpleAstroFilter() {
    this->Release();
}

/*
 * SimpleAstroFilter::create
 */
bool SimpleAstroFilter::create() {
    // intentionally empty
    return true;
}

/*
 * SimpleAstroFilter::release
 */
void SimpleAstroFilter::release() {
    // intentionally empty
}

/*
 * SimpleAstroFilter::getData
 */
bool SimpleAstroFilter::getData(core::Call& call) {
    AstroDataCall* adc = dynamic_cast<AstroDataCall*>(&call);
    if (adc == nullptr)
        return false;

    AstroDataCall* inCall = this->particlesInSlot.CallAs<AstroDataCall>();
    if (inCall == nullptr)
        return false;

    inCall->operator=(*adc);
    inCall->SetUnlocker(nullptr, false);
    if ((*inCall)(AstroDataCall::CallForGetData)) {
        if (this->refilter) {
            this->filter(*inCall);
            this->refilter = false;
        }
        this->copyContentToOutCall(*adc);
        if (this->fillFilterButtonParam.IsDirty()) {
            this->setDisplayedValues(*adc);
            this->fillFilterButtonParam.ResetDirty();
        }
        inCall->Unlock();
        return true;
    }
    inCall->Unlock();
    return false;
}

/*
 * SimpleAstroFilter::getExtent
 */
bool SimpleAstroFilter::getExtent(core::Call& call) {
    AstroDataCall* adc = dynamic_cast<AstroDataCall*>(&call);
    if (adc == nullptr)
        return false;

    AstroDataCall* inCall = this->particlesInSlot.CallAs<AstroDataCall>();
    if (inCall == nullptr)
        return false;

    inCall->operator=(*adc);
    adc->SetUnlocker(nullptr, false);
    if ((*inCall)(AstroDataCall::CallForGetExtent)) {
        adc->operator=(*inCall);
        if (this->lastDataHash != inCall->DataHash() || this->lastTimestep != adc->FrameID() || this->isParamDirty()) {
            this->hashOffset++;
            this->lastTimestep = adc->FrameID();
            this->lastDataHash = inCall->DataHash();
            this->refilter = true;
            this->resetDirtyParams();
        }
        adc->SetDataHash(this->lastDataHash + this->hashOffset);
        return true;
    }
    return false;
}

/*
 * SimpleAstroFilter::initFields
 */
void SimpleAstroFilter::initFields() {
    if (this->positions == nullptr) {
        this->positions = std::make_shared<std::vector<glm::vec3>>();
    }
    if (this->velocities == nullptr) {
        this->velocities = std::make_shared<std::vector<glm::vec3>>();
    }
    if (this->temperatures == nullptr) {
        this->temperatures = std::make_shared<std::vector<float>>();
    }
    if (this->masses == nullptr) {
        this->masses = std::make_shared<std::vector<float>>();
    }
    if (this->internalEnergies == nullptr) {
        this->internalEnergies = std::make_shared<std::vector<float>>();
    }
    if (this->smoothingLengths == nullptr) {
        this->smoothingLengths = std::make_shared<std::vector<float>>();
    }
    if (this->molecularWeights == nullptr) {
        this->molecularWeights = std::make_shared<std::vector<float>>();
    }
    if (this->densities == nullptr) {
        this->densities = std::make_shared<std::vector<float>>();
    }
    if (this->gravitationalPotentials == nullptr) {
        this->gravitationalPotentials = std::make_shared<std::vector<float>>();
    }
    if (this->entropies == nullptr) {
        this->entropies = std::make_shared<std::vector<float>>();
    }
    if (this->isBaryonFlags == nullptr) {
        this->isBaryonFlags = std::make_shared<std::vector<bool>>();
    }
    if (this->isStarFlags == nullptr) {
        this->isStarFlags = std::make_shared<std::vector<bool>>();
    }
    if (this->isWindFlags == nullptr) {
        this->isWindFlags = std::make_shared<std::vector<bool>>();
    }
    if (this->isStarFormingGasFlags == nullptr) {
        this->isStarFormingGasFlags = std::make_shared<std::vector<bool>>();
    }
    if (this->isAGNFlags == nullptr) {
        this->isAGNFlags = std::make_shared<std::vector<bool>>();
    }
    if (this->particleIDs == nullptr) {
        this->particleIDs = std::make_shared<std::vector<int64_t>>();
    }
    if (this->agnDistances == nullptr) {
        this->agnDistances = std::make_shared<std::vector<float>>();
    }
}

/*
 * SimpleAstroFilter::filter
 */
bool SimpleAstroFilter::filter(const AstroDataCall& call) {
    std::set<uint64_t> filterResult;
    std::vector<uint64_t> help(call.GetParticleCount());
    std::iota(help.begin(), help.end(), 0);
    filterResult.insert(help.begin(), help.end());

    for (uint64_t i = 0; i < call.GetParticleCount(); ++i) {
        if (this->showOnlyBaryonParam.Param<param::BoolParam>()->Value() && !call.GetIsBaryonFlags()->at(i)) {
            filterResult.erase(i);
        }
        if (this->showOnlyDarkMatterParam.Param<param::BoolParam>()->Value() && call.GetIsBaryonFlags()->at(i)) {
            filterResult.erase(i);
        }
        if (this->showOnlyStarsParam.Param<param::BoolParam>()->Value() && !call.GetIsStarFlags()->at(i)) {
            filterResult.erase(i);
        }
        if (this->showOnlyWindParam.Param<param::BoolParam>()->Value() && !call.GetIsWindFlags()->at(i)) {
            filterResult.erase(i);
        }
        if (this->showOnlyStarFormingGasParam.Param<param::BoolParam>()->Value() &&
            !call.GetIsStarFormingGasFlags()->at(i)) {
            filterResult.erase(i);
        }
        if (this->showOnlyAGNsParam.Param<param::BoolParam>()->Value() && !call.GetIsAGNFlags()->at(i)) {
            filterResult.erase(i);
        }
        if (this->filterVelocityMagnitudeParam.Param<param::BoolParam>()->Value()) {
            if (glm::length(call.GetVelocities()->at(i)) <
                    this->minVelocityMagnitudeParam.Param<param::FloatParam>()->Value() ||
                glm::length(call.GetVelocities()->at(i)) >
                    this->maxVelocityMagnitudeParam.Param<param::FloatParam>()->Value()) {
                filterResult.erase(i);
            }
        }
        if (this->filterTemperatureParam.Param<param::BoolParam>()->Value()) {
            if (call.GetTemperature()->at(i) < this->minTemperatureParam.Param<param::FloatParam>()->Value() ||
                call.GetTemperature()->at(i) > this->maxTemperatureParam.Param<param::FloatParam>()->Value()) {
                filterResult.erase(i);
            }
        }
        if (this->filterMassParam.Param<param::BoolParam>()->Value()) {
            if (call.GetMass()->at(i) < this->minMassParam.Param<param::FloatParam>()->Value() ||
                call.GetMass()->at(i) > this->maxMassParam.Param<param::FloatParam>()->Value()) {
                filterResult.erase(i);
            }
        }
        if (this->filterInternalEnergyParam.Param<param::BoolParam>()->Value()) {
            if (call.GetInternalEnergy()->at(i) < this->minInternalEnergyParam.Param<param::FloatParam>()->Value() ||
                call.GetInternalEnergy()->at(i) > this->maxInternalEnergyParam.Param<param::FloatParam>()->Value()) {
                filterResult.erase(i);
            }
        }
        if (this->filterSmoothingLengthParam.Param<param::BoolParam>()->Value()) {
            if (call.GetSmoothingLength()->at(i) < this->minSmoothingLengthParam.Param<param::FloatParam>()->Value() ||
                call.GetSmoothingLength()->at(i) > this->maxSmoothingLengthParam.Param<param::FloatParam>()->Value()) {
                filterResult.erase(i);
            }
        }
        if (this->filterMolecularWeightParam.Param<param::BoolParam>()->Value()) {
            if (call.GetMolecularWeights()->at(i) < this->minMolecularWeightParam.Param<param::FloatParam>()->Value() ||
                call.GetMolecularWeights()->at(i) > this->maxMolecularWeightParam.Param<param::FloatParam>()->Value()) {
                filterResult.erase(i);
            }
        }
        if (this->filterDensityParam.Param<param::BoolParam>()->Value()) {
            if (call.GetDensity()->at(i) < this->minDensityParam.Param<param::FloatParam>()->Value() ||
                call.GetDensity()->at(i) > this->maxDensityParam.Param<param::FloatParam>()->Value()) {
                filterResult.erase(i);
            }
        }
        if (this->filterGravitationalPotentialParam.Param<param::BoolParam>()->Value()) {
            if (call.GetGravitationalPotential()->at(i) <
                    this->minGravitationalPotentialParam.Param<param::FloatParam>()->Value() ||
                call.GetGravitationalPotential()->at(i) >
                    this->maxGravitationalPotentialParam.Param<param::FloatParam>()->Value()) {
                filterResult.erase(i);
            }
        }
        if (this->filterEntropyParam.Param<param::BoolParam>()->Value()) {
            if (call.GetEntropy()->at(i) < this->minEntropyParam.Param<param::FloatParam>()->Value() ||
                call.GetEntropy()->at(i) > this->maxEntropyParam.Param<param::FloatParam>()->Value()) {
                filterResult.erase(i);
            }
        }
        if (this->filterAgnDistanceParam.Param<param::BoolParam>()->Value()) {
            if (call.GetAgnDistances()->at(i) < this->minAgnDistanceParam.Param<param::FloatParam>()->Value() ||
                call.GetAgnDistances()->at(i) > this->maxAgnDistanceParam.Param<param::FloatParam>()->Value()) {
                filterResult.erase(i);
            }
        }
    }
    return this->copyInCallToContent(call, filterResult);
}

/*
 * SimpleAstroFilter::copyContentToOutCall
 */
bool SimpleAstroFilter::copyContentToOutCall(AstroDataCall& outCall) {
    outCall.SetPositions(this->positions);
    outCall.SetVelocities(this->velocities);
    outCall.SetTemperature(this->temperatures);
    outCall.SetMass(this->masses);
    outCall.SetInternalEnergy(this->internalEnergies);
    outCall.SetSmoothingLength(this->smoothingLengths);
    outCall.SetMolecularWeights(this->molecularWeights);
    outCall.SetDensity(this->densities);
    outCall.SetGravitationalPotential(this->gravitationalPotentials);
    outCall.SetEntropy(this->entropies);
    outCall.SetIsBaryonFlags(this->isBaryonFlags);
    outCall.SetIsStarFlags(this->isStarFlags);
    outCall.SetIsWindFlags(this->isWindFlags);
    outCall.SetIsStarFormingGasFlags(this->isStarFormingGasFlags);
    outCall.SetIsAGNFlags(this->isAGNFlags);
    outCall.SetParticleIDs(this->particleIDs);
    outCall.SetAGNDistances(this->agnDistances);
    return true;
}

/*
 * SimpleAstroFilter::copyInCallToContent
 */
bool SimpleAstroFilter::copyInCallToContent(const AstroDataCall& inCall, const std::set<uint64_t>& indexSet) {
    this->positions->resize(indexSet.size());
    this->velocities->resize(indexSet.size());
    this->temperatures->resize(indexSet.size());
    this->masses->resize(indexSet.size());
    this->internalEnergies->resize(indexSet.size());
    this->smoothingLengths->resize(indexSet.size());
    this->molecularWeights->resize(indexSet.size());
    this->densities->resize(indexSet.size());
    this->gravitationalPotentials->resize(indexSet.size());
    this->entropies->resize(indexSet.size());
    this->isBaryonFlags->resize(indexSet.size());
    this->isStarFlags->resize(indexSet.size());
    this->isWindFlags->resize(indexSet.size());
    this->isStarFormingGasFlags->resize(indexSet.size());
    this->isAGNFlags->resize(indexSet.size());
    this->particleIDs->resize(indexSet.size());
    this->agnDistances->resize(indexSet.size());

    std::vector<uint64_t> setVec(indexSet.begin(), indexSet.end());
    std::sort(setVec.begin(), setVec.end());

    uint64_t i = 0;
    for (const auto id : setVec) {
        this->positions->at(i) = inCall.GetPositions()->at(id);
        this->velocities->at(i) = inCall.GetVelocities()->at(id);
        this->temperatures->at(i) = inCall.GetTemperature()->at(id);
        this->masses->at(i) = inCall.GetMass()->at(id);
        this->internalEnergies->at(i) = inCall.GetInternalEnergy()->at(id);
        this->smoothingLengths->at(i) = inCall.GetSmoothingLength()->at(id);
        this->molecularWeights->at(i) = inCall.GetMolecularWeights()->at(id);
        this->densities->at(i) = inCall.GetDensity()->at(id);
        this->gravitationalPotentials->at(i) = inCall.GetGravitationalPotential()->at(id);
        this->entropies->at(i) = inCall.GetEntropy()->at(i);
        this->isBaryonFlags->at(i) = inCall.GetIsBaryonFlags()->at(id);
        this->isStarFlags->at(i) = inCall.GetIsStarFlags()->at(id);
        this->isWindFlags->at(i) = inCall.GetIsWindFlags()->at(id);
        this->isStarFormingGasFlags->at(i) = inCall.GetIsStarFormingGasFlags()->at(id);
        this->isAGNFlags->at(i) = inCall.GetIsAGNFlags()->at(id);
        this->particleIDs->at(i) = inCall.GetParticleIDs()->at(id);
        this->agnDistances->at(i) = inCall.GetAgnDistances()->at(id);
        ++i;
    }
    return true;
}

/*
 * SimpleAstroFilter::isParamDirty
 */
bool SimpleAstroFilter::isParamDirty() {
    if (this->showOnlyBaryonParam.IsDirty())
        return true;
    if (this->showOnlyDarkMatterParam.IsDirty())
        return true;
    if (this->showOnlyStarsParam.IsDirty())
        return true;
    if (this->showOnlyWindParam.IsDirty())
        return true;
    if (this->showOnlyStarFormingGasParam.IsDirty())
        return true;
    if (this->showOnlyAGNsParam.IsDirty())
        return true;
    if (this->minVelocityMagnitudeParam.IsDirty())
        return true;
    if (this->maxVelocityMagnitudeParam.IsDirty())
        return true;
    if (this->filterVelocityMagnitudeParam.IsDirty())
        return true;
    if (this->minTemperatureParam.IsDirty())
        return true;
    if (this->maxTemperatureParam.IsDirty())
        return true;
    if (this->filterTemperatureParam.IsDirty())
        return true;
    if (this->minMassParam.IsDirty())
        return true;
    if (this->maxMassParam.IsDirty())
        return true;
    if (this->filterMassParam.IsDirty())
        return true;
    if (this->minInternalEnergyParam.IsDirty())
        return true;
    if (this->maxInternalEnergyParam.IsDirty())
        return true;
    if (this->filterInternalEnergyParam.IsDirty())
        return true;
    if (this->minSmoothingLengthParam.IsDirty())
        return true;
    if (this->maxSmoothingLengthParam.IsDirty())
        return true;
    if (this->filterSmoothingLengthParam.IsDirty())
        return true;
    if (this->minMolecularWeightParam.IsDirty())
        return true;
    if (this->maxMolecularWeightParam.IsDirty())
        return true;
    if (this->filterMolecularWeightParam.IsDirty())
        return true;
    if (this->minDensityParam.IsDirty())
        return true;
    if (this->maxDensityParam.IsDirty())
        return true;
    if (this->filterDensityParam.IsDirty())
        return true;
    if (this->minGravitationalPotentialParam.IsDirty())
        return true;
    if (this->maxGravitationalPotentialParam.IsDirty())
        return true;
    if (this->filterGravitationalPotentialParam.IsDirty())
        return true;
    if (this->minEntropyParam.IsDirty())
        return true;
    if (this->maxEntropyParam.IsDirty())
        return true;
    if (this->filterEntropyParam.IsDirty())
        return true;
    if (this->minAgnDistanceParam.IsDirty())
        return true;
    if (this->maxAgnDistanceParam.IsDirty())
        return true;
    if (this->filterAgnDistanceParam.IsDirty())
        return true;
    return false;
}

/*
 * SimpleAstroFilter::resetDirtyParams
 */
void SimpleAstroFilter::resetDirtyParams() {
    this->showOnlyBaryonParam.ResetDirty();
    this->showOnlyDarkMatterParam.ResetDirty();
    this->showOnlyStarsParam.ResetDirty();
    this->showOnlyWindParam.ResetDirty();
    this->showOnlyStarFormingGasParam.ResetDirty();
    this->showOnlyAGNsParam.ResetDirty();
    this->minVelocityMagnitudeParam.ResetDirty();
    this->maxVelocityMagnitudeParam.ResetDirty();
    this->filterVelocityMagnitudeParam.ResetDirty();
    this->minTemperatureParam.ResetDirty();
    this->maxTemperatureParam.ResetDirty();
    this->filterTemperatureParam.ResetDirty();
    this->minMassParam.ResetDirty();
    this->maxMassParam.ResetDirty();
    this->filterMassParam.ResetDirty();
    this->minInternalEnergyParam.ResetDirty();
    this->maxInternalEnergyParam.ResetDirty();
    this->filterInternalEnergyParam.ResetDirty();
    this->minSmoothingLengthParam.ResetDirty();
    this->maxSmoothingLengthParam.ResetDirty();
    this->filterSmoothingLengthParam.ResetDirty();
    this->minMolecularWeightParam.ResetDirty();
    this->maxMolecularWeightParam.ResetDirty();
    this->filterMolecularWeightParam.ResetDirty();
    this->minDensityParam.ResetDirty();
    this->maxDensityParam.ResetDirty();
    this->filterDensityParam.ResetDirty();
    this->minGravitationalPotentialParam.ResetDirty();
    this->maxGravitationalPotentialParam.ResetDirty();
    this->filterGravitationalPotentialParam.ResetDirty();
    this->minEntropyParam.ResetDirty();
    this->maxEntropyParam.ResetDirty();
    this->filterEntropyParam.ResetDirty();
    this->minAgnDistanceParam.ResetDirty();
    this->maxAgnDistanceParam.ResetDirty();
    this->filterAgnDistanceParam.ResetDirty();
}

/*
 * SimpleAstroFilter::setDisplayedValues
 */
void SimpleAstroFilter::setDisplayedValues(const AstroDataCall& outCall) {
    float minVelocity = FLT_MAX, maxVelocity = -FLT_MAX;
    float minTemperature = FLT_MAX, maxTemperature = -FLT_MAX;
    float minMass = FLT_MAX, maxMass = -FLT_MAX;
    float minInternalEnergy = FLT_MAX, maxInternalEnergy = -FLT_MAX;
    float minSmoothingLength = FLT_MAX, maxSmoothingLength = -FLT_MAX;
    float minMolecularWeight = FLT_MAX, maxMolecularWeight = -FLT_MAX;
    float minDensity = FLT_MAX, maxDensity = -FLT_MAX;
    float minGravitationalPotential = FLT_MAX, maxGravitationalPotential = -FLT_MAX;
    float minEntropy = FLT_MAX, maxEntropy = -FLT_MAX;
    float minAGNDistance = FLT_MAX, maxAGNDistance = -FLT_MAX;

    for (uint64_t i = 0; i < outCall.GetParticleCount(); ++i) {
        if (glm::length(outCall.GetVelocities()->at(i)) < minVelocity) {
            minVelocity = glm::length(outCall.GetVelocities()->at(i));
        }
        if (glm::length(outCall.GetVelocities()->at(i)) > maxVelocity) {
            maxVelocity = glm::length(outCall.GetVelocities()->at(i));
        }

        if (outCall.GetTemperature()->at(i) < minTemperature) {
            minTemperature = outCall.GetTemperature()->at(i);
        }
        if (outCall.GetTemperature()->at(i) > maxTemperature) {
            maxTemperature = outCall.GetTemperature()->at(i);
        }

        if (outCall.GetMass()->at(i) < minMass) {
            minMass = outCall.GetMass()->at(i);
        }
        if (outCall.GetMass()->at(i) > maxMass) {
            maxMass = outCall.GetMass()->at(i);
        }

        if (outCall.GetInternalEnergy()->at(i) < minInternalEnergy) {
            minInternalEnergy = outCall.GetInternalEnergy()->at(i);
        }
        if (outCall.GetInternalEnergy()->at(i) > maxInternalEnergy) {
            maxInternalEnergy = outCall.GetInternalEnergy()->at(i);
        }

        if (outCall.GetSmoothingLength()->at(i) < minSmoothingLength) {
            minSmoothingLength = outCall.GetSmoothingLength()->at(i);
        }
        if (outCall.GetSmoothingLength()->at(i) > maxSmoothingLength) {
            maxSmoothingLength = outCall.GetSmoothingLength()->at(i);
        }

        if (outCall.GetMolecularWeights()->at(i) < minMolecularWeight) {
            minMolecularWeight = outCall.GetMolecularWeights()->at(i);
        }
        if (outCall.GetMolecularWeights()->at(i) > maxMolecularWeight) {
            maxMolecularWeight = outCall.GetMolecularWeights()->at(i);
        }

        if (outCall.GetDensity()->at(i) < minDensity) {
            minDensity = outCall.GetDensity()->at(i);
        }
        if (outCall.GetDensity()->at(i) > maxDensity) {
            maxDensity = outCall.GetDensity()->at(i);
        }

        if (outCall.GetGravitationalPotential()->at(i) < minGravitationalPotential) {
            minGravitationalPotential = outCall.GetGravitationalPotential()->at(i);
        }
        if (outCall.GetGravitationalPotential()->at(i) > maxGravitationalPotential) {
            maxGravitationalPotential = outCall.GetGravitationalPotential()->at(i);
        }

        if (outCall.GetEntropy()->at(i) < minEntropy) {
            minEntropy = outCall.GetEntropy()->at(i);
        }
        if (outCall.GetEntropy()->at(i) > maxEntropy) {
            maxEntropy = outCall.GetEntropy()->at(i);
        }

        if (outCall.GetAgnDistances()->at(i) < minAGNDistance) {
            minAGNDistance = outCall.GetAgnDistances()->at(i);
        }
        if (outCall.GetAgnDistances()->at(i) > maxAGNDistance) {
            maxAGNDistance = outCall.GetAgnDistances()->at(i);
        }
    }

    this->minVelocityMagnitudeParam.Param<param::FloatParam>()->SetValue(minVelocity, false);
    this->maxVelocityMagnitudeParam.Param<param::FloatParam>()->SetValue(maxVelocity, false);
    this->minTemperatureParam.Param<param::FloatParam>()->SetValue(minTemperature, false);
    this->maxTemperatureParam.Param<param::FloatParam>()->SetValue(maxTemperature, false);
    this->minMassParam.Param<param::FloatParam>()->SetValue(minMass, false);
    this->maxMassParam.Param<param::FloatParam>()->SetValue(maxMass, false);
    this->minInternalEnergyParam.Param<param::FloatParam>()->SetValue(minInternalEnergy, false);
    this->maxInternalEnergyParam.Param<param::FloatParam>()->SetValue(maxInternalEnergy, false);
    this->minSmoothingLengthParam.Param<param::FloatParam>()->SetValue(minSmoothingLength, false);
    this->maxSmoothingLengthParam.Param<param::FloatParam>()->SetValue(maxSmoothingLength, false);
    this->minMolecularWeightParam.Param<param::FloatParam>()->SetValue(minMolecularWeight, false);
    this->maxMolecularWeightParam.Param<param::FloatParam>()->SetValue(maxMolecularWeight, false);
    this->minDensityParam.Param<param::FloatParam>()->SetValue(minDensity, false);
    this->maxDensityParam.Param<param::FloatParam>()->SetValue(maxDensity, false);
    this->minGravitationalPotentialParam.Param<param::FloatParam>()->SetValue(minGravitationalPotential, false);
    this->maxGravitationalPotentialParam.Param<param::FloatParam>()->SetValue(maxGravitationalPotential, false);
    this->minEntropyParam.Param<param::FloatParam>()->SetValue(minEntropy, false);
    this->maxEntropyParam.Param<param::FloatParam>()->SetValue(maxEntropy, false);
    this->minAgnDistanceParam.Param<param::FloatParam>()->SetValue(minAGNDistance, false);
    this->maxAgnDistanceParam.Param<param::FloatParam>()->SetValue(maxAGNDistance, false);
}
