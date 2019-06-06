/*
 * FilamentFilter.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "FilamentFilter.h"

#include <algorithm>
#include <climits>
#include <set>
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"

using namespace megamol;
using namespace megamol::astro;
using namespace megamol::core;

/*
 * FilamentFilter::FilamentFilter
 */
FilamentFilter::FilamentFilter(void)
    : Module()
    , filamentOutSlot("filamentOut", "Output slot for the filament particles")
    , particlesInSlot("particlesIn", "Input slot for the astro particle data")
    , radiusSlot("radius", "The used radius for the FOF algorithm")
    , isActiveSlot("isActive", "When deactivated this module only passes through the incoming data")
    , densitySeedPercentageSlot("densityPercentage", "Percentag of seed densities over all data points")
    , recalculateFilaments(true)
    , hashOffset(0)
    , lastDataHash(0) {

    this->particlesInSlot.SetCompatibleCall<AstroDataCallDescription>();
    this->MakeSlotAvailable(&this->particlesInSlot);

    this->filamentOutSlot.SetCallback(AstroDataCall::ClassName(),
        AstroDataCall::FunctionName(AstroDataCall::CallForGetData), &FilamentFilter::getData);
    this->filamentOutSlot.SetCallback(AstroDataCall::ClassName(),
        AstroDataCall::FunctionName(AstroDataCall::CallForGetExtent), &FilamentFilter::getExtent);
    this->MakeSlotAvailable(&this->filamentOutSlot);

    this->isActiveSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->isActiveSlot);

    this->radiusSlot.SetParameter(new param::FloatParam(0.1f, 0.0f));
    this->MakeSlotAvailable(&this->radiusSlot);

    this->densitySeedPercentageSlot.SetParameter(new param::FloatParam(10.0f, 0.0f, 100.0f));
    this->MakeSlotAvailable(&this->densitySeedPercentageSlot);

    this->initFields();
}

/*
 * FilamentFilter::~FilamentFilter
 */
FilamentFilter::~FilamentFilter(void) { this->Release(); }

/*
 * FilamentFilter::create
 */
bool FilamentFilter::create(void) {
    // intentionally empty
    return true;
}

/*
 * FilamentFilter::release
 */
void FilamentFilter::release(void) {
    // intentionally empty
}

/*
 * FilamentFilter::getData
 */
bool FilamentFilter::getData(core::Call& call) {
    AstroDataCall* adc = dynamic_cast<AstroDataCall*>(&call);
    if (adc == nullptr) return false;

    AstroDataCall* inCall = this->particlesInSlot.CallAs<AstroDataCall>();
    if (inCall == nullptr) return false;

    inCall->operator=(*adc);
    if ((*inCall)(AstroDataCall::CallForGetData)) {
        if (this->isActiveSlot.Param<param::BoolParam>()->Value()) {
            if (this->recalculateFilaments) {
                this->filterFilaments(*inCall);
                this->recalculateFilaments = false;
            }
            this->copyContentToOutCall(*adc);
        } else {
            adc->operator=(*inCall);
        }
        return true;
    }
    return false;
}

/*
 * FilamentFilter::getExtent
 */
bool FilamentFilter::getExtent(core::Call& call) {
    AstroDataCall* adc = dynamic_cast<AstroDataCall*>(&call);
    if (adc == nullptr) return false;

    AstroDataCall* inCall = this->particlesInSlot.CallAs<AstroDataCall>();
    if (inCall == nullptr) return false;

    inCall->operator=(*adc);
    if ((*inCall)(AstroDataCall::CallForGetExtent)) {
        adc->operator=(*inCall);
        if (this->lastDataHash != inCall->DataHash() || this->radiusSlot.IsDirty() ||
            this->densitySeedPercentageSlot.IsDirty()) {
            this->lastDataHash = inCall->DataHash();
            this->radiusSlot.ResetDirty();
            this->densitySeedPercentageSlot.ResetDirty();
            this->recalculateFilaments = true;
        }
        adc->SetDataHash(this->lastDataHash + this->hashOffset);
        return true;
    }
    return false;
}

/*
 * FilamentFilter::copyContentToOutCall
 */
bool FilamentFilter::copyContentToOutCall(AstroDataCall& outCall) {
    outCall.SetPositions(this->positions);
    outCall.SetVelocities(this->velocities);
    outCall.SetTemperature(this->temperatures);
    outCall.SetMass(this->masses);
    outCall.SetInternalEnergy(this->internalEnergies);
    outCall.SetSmoothingLength(this->smoothingLengths);
    outCall.SetMolecularWeights(this->molecularWeights);
    outCall.SetDensity(this->densities);
    outCall.SetGravitationalPotential(this->gravitationalPotentials);
    outCall.SetIsBaryonFlags(this->isBaryonFlags);
    outCall.SetIsStarFlags(this->isStarFlags);
    outCall.SetIsWindFlags(this->isWindFlags);
    outCall.SetIsStarFormingGasFlags(this->isStarFormingGasFlags);
    outCall.SetIsAGNFlags(this->isAGNFlags);
    outCall.SetParticleIDs(this->particleIDs);
    return true;
}

/*
 * FilamentFilter::initFields
 */
void FilamentFilter::initFields(void) {
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
}

/*
 * FilamentFilter::getMinMaxDensity
 */
std::pair<float, float> FilamentFilter::getMinMaxDensity(const AstroDataCall& call) const {
    const auto& dens = call.GetDensity();
    if (dens == nullptr) return std::make_pair(0.0f, 0.0f);
    auto resit = std::minmax_element(dens->begin(), dens->end());
    return std::make_pair(*resit.first, *resit.second);
}

/*
 * FilamentFilter::getMinMaxDensity
 */
void FilamentFilter::retrieveDensityCandidateList(
    const AstroDataCall& call, std::vector<std::pair<float, uint64_t>>& result) {
    result.clear();
    const auto& dens = call.GetDensity();
    if (dens == nullptr) return;
    auto minmax = this->getMinMaxDensity(call);
    result.resize(dens->size());
    for (uint64_t i = 0; i < dens->size(); i++) {
        result[i] = std::make_pair(dens->at(i), i);
    }
    // sort all the densities in descending order and only keep a certain percentage
    std::sort(result.rbegin(), result.rend());
    float percentage = this->densitySeedPercentageSlot.Param<param::FloatParam>()->Value();
    percentage /= 100.0f;
    float minDensity = percentage * minmax.second;
    auto foundval =
        std::find_if(result.begin(), result.end(), [&minDensity](const auto& x) { return minDensity > x.first; });
    result.erase(foundval, result.end());
}

/*
 * FilamentFilter::initSearchStructure
 */
void FilamentFilter::initSearchStructure(const AstroDataCall& call) {
    const auto& posPtr = call.GetPositions();
    this->pointCloud.pts.resize(posPtr->size());
    std::memcpy(this->pointCloud.pts.data(), posPtr->data(), posPtr->size() * sizeof(glm::vec3));
    if (this->searchIndexPtr != nullptr) {
        this->searchIndexPtr.reset();
    }
    this->searchIndexPtr =
        std::make_shared<my_kd_tree_t>(3, this->pointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    this->searchIndexPtr->buildIndex();
}

/*
 * FilamentFilter::filterFilaments
 */
bool FilamentFilter::filterFilaments(const AstroDataCall& call) {
    if (call.GetPositions() == nullptr) return false;
    std::vector<std::pair<float, uint64_t>> densityPeaks;
    this->retrieveDensityCandidateList(call, densityPeaks);
    this->initSearchStructure(call);
    if (this->searchIndexPtr == nullptr) return false;
    // the following approach is not really performant, but it should work
    std::vector<std::set<uint64_t>> setVec(densityPeaks.size());
    std::vector<size_t> vertSetIndex(call.GetPositions()->size());
    std::vector<bool> calculatedFlags(call.GetPositions()->size(), false);
    for (size_t i = 0; i < setVec.size(); ++i) {
        setVec[i].insert(densityPeaks[i].second);
        vertSetIndex[densityPeaks[i].second] = i;
    }
    // TODO insert elements into working queue, determine sets to insert into, ...
    return true;
}
