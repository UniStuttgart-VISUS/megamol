/*
 * FilamentFilter.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "FilamentFilter.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include <algorithm>
#include <climits>
#include <queue>
#include <set>

using namespace megamol;
using namespace megamol::astro;
using namespace megamol::core;

/*
 * FilamentFilter::FilamentFilter
 */
FilamentFilter::FilamentFilter()
        : Module()
        , filamentOutSlot("filamentOut", "Output slot for the filament particles")
        , particlesInSlot("particlesIn", "Input slot for the astro particle data")
        , radiusSlot("radius", "The used radius for the FOF algorithm")
        , isActiveSlot("isActive", "When deactivated this module only passes through the incoming data")
        , densitySeedPercentageSlot(
              "densityPercentage", "Percentage of data points that is thrown away because of too low density")
        , minClusterSizeSlot("minClusterSize", "Minimal number of particles in a detected cluster")
        , maxParticlePercentageCuttoff(
              "maxParticlePercentage", "Maximum percentage of particles that is considered as candidates")
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

    this->radiusSlot.SetParameter(new param::FloatParam(0.45f, 0.0f));
    this->MakeSlotAvailable(&this->radiusSlot);

    this->minClusterSizeSlot.SetParameter(new param::IntParam(100, 2));
    this->MakeSlotAvailable(&this->minClusterSizeSlot);

    this->densitySeedPercentageSlot.SetParameter(new param::FloatParam(90.0f, 0.0f, 100.0f));
    this->MakeSlotAvailable(&this->densitySeedPercentageSlot);

    this->maxParticlePercentageCuttoff.SetParameter(new param::FloatParam(1.0f, 0.0f, 100.0f));
    this->MakeSlotAvailable(&this->maxParticlePercentageCuttoff);

    this->initFields();
}

/*
 * FilamentFilter::~FilamentFilter
 */
FilamentFilter::~FilamentFilter() {
    this->Release();
}

/*
 * FilamentFilter::create
 */
bool FilamentFilter::create() {
    // intentionally empty
    return true;
}

/*
 * FilamentFilter::release
 */
void FilamentFilter::release() {
    // intentionally empty
}

/*
 * FilamentFilter::getData
 */
bool FilamentFilter::getData(core::Call& call) {
    AstroDataCall* adc = dynamic_cast<AstroDataCall*>(&call);
    if (adc == nullptr)
        return false;

    AstroDataCall* inCall = this->particlesInSlot.CallAs<AstroDataCall>();
    if (inCall == nullptr)
        return false;

    inCall->operator=(*adc);
    inCall->SetUnlocker(nullptr, false);
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
        inCall->Unlock();
        return true;
    }
    inCall->Unlock();
    return false;
}

/*
 * FilamentFilter::getExtent
 */
bool FilamentFilter::getExtent(core::Call& call) {
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
        if (this->lastDataHash != inCall->DataHash() || this->lastTimestep != adc->FrameID() ||
            this->radiusSlot.IsDirty() || this->densitySeedPercentageSlot.IsDirty() ||
            this->minClusterSizeSlot.IsDirty() || this->maxParticlePercentageCuttoff.IsDirty()) {
            this->hashOffset++;
            this->lastTimestep = adc->FrameID();
            this->lastDataHash = inCall->DataHash();
            this->radiusSlot.ResetDirty();
            this->densitySeedPercentageSlot.ResetDirty();
            this->minClusterSizeSlot.ResetDirty();
            this->maxParticlePercentageCuttoff.ResetDirty();
            this->recalculateFilaments = true;
        }
        if (this->isActiveSlot.IsDirty() && this->positions != nullptr && !this->positions->empty()) {
            this->isActiveSlot.ResetDirty();
            this->recalculateFilaments = true;
            this->hashOffset++;
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
    outCall.SetEntropy(this->entropies);
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
void FilamentFilter::initFields() {
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
}

/*
 * FilamentFilter::getMinMaxDensity
 */
std::pair<float, float> FilamentFilter::getMinMaxDensity(const AstroDataCall& call) const {
    const auto& dens = call.GetDensity();
    if (dens == nullptr)
        return std::make_pair(0.0f, 0.0f);
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
    if (dens == nullptr)
        return;
    auto minmax = this->getMinMaxDensity(call);
    result.resize(dens->size());
    for (uint64_t i = 0; i < dens->size(); i++) {
        result[i] = std::make_pair(dens->at(i), i);
    }
    // sort all the densities in descending order and only keep a certain percentage
    std::sort(result.rbegin(), result.rend());
    float percentage = this->densitySeedPercentageSlot.Param<param::FloatParam>()->Value();
    percentage = 100.0f - percentage;
    percentage /= 100.0f;
    float minDensity = percentage * minmax.second;
    auto foundval =
        std::find_if(result.begin(), result.end(), [&minDensity](const auto& x) { return minDensity > x.first; });
    result.erase(foundval, result.end());
    const auto maxPartCount = static_cast<uint64_t>(
        call.GetParticleCount() * (this->maxParticlePercentageCuttoff.Param<param::FloatParam>()->Value() / 100.0f));
    if (result.size() > maxPartCount) {
        result.resize(maxPartCount);
    }
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
 * FilamentFilter::copyInCallToContent
 */
bool FilamentFilter::copyInCallToContent(const AstroDataCall& inCall, const std::set<uint64_t>& indexSet) {
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
        ++i;
    }
    return true;
}

/*
 * FilamentFilter::filterFilaments
 */
bool FilamentFilter::filterFilaments(const AstroDataCall& call) {
    if (call.GetPositions() == nullptr)
        return false;
    std::vector<std::pair<float, uint64_t>> densityPeaks;
    this->retrieveDensityCandidateList(call, densityPeaks);
    std::set<uint64_t> candidateSet;
    for (const auto& a : densityPeaks) {
        candidateSet.insert(a.second);
    }
    this->initSearchStructure(call);
    if (this->searchIndexPtr == nullptr)
        return false;

    // the following approach is not really performant, but it should work
    std::vector<std::set<uint64_t>> setVec;
    std::vector<bool> calculatedFlags(call.GetPositions()->size(), false);

    nanoflann::SearchParameters searchParams;
    float searchRadius = this->radiusSlot.Param<param::FloatParam>()->Value();
    std::vector<nanoflann::ResultItem<size_t, float>> searchResults;
    std::set<uint64_t> toProcessSet;

    while (!candidateSet.empty()) {
        auto current = *candidateSet.begin();
        auto position = call.GetPositions()->at(current);
        setVec.push_back(std::set<uint64_t>());
        setVec.back().insert(current);
        toProcessSet.clear();
        calculatedFlags[current] = true;
        const auto nMatches =
            this->searchIndexPtr->radiusSearch(&position.x, searchRadius * searchRadius, searchResults, searchParams);
        // insert everything in the vicinity of the current candidate to the queue
        for (const auto& v : searchResults) {
            uint64_t index = v.first;
            if (index == current)
                continue;
            if (!calculatedFlags[index]) {
                toProcessSet.insert(index);
                setVec.back().insert(index);
                calculatedFlags[index] = true;
            }
        }
        while (!toProcessSet.empty()) {
            auto cur = *toProcessSet.begin();
            auto pos = call.GetPositions()->at(cur);
            searchResults.clear();
            const auto matches =
                this->searchIndexPtr->radiusSearch(&pos.x, searchRadius * searchRadius, searchResults, searchParams);
            for (const auto& v : searchResults) {
                uint64_t index = v.first;
                if (index == cur)
                    continue;
                if (!calculatedFlags[index]) {
                    toProcessSet.insert(index);
                    setVec.back().insert(index);
                    calculatedFlags[index] = true;
                }
            }
            toProcessSet.erase(cur);
            candidateSet.erase(cur);
        }
        candidateSet.erase(current);
    }
    // erase too small clusters
    auto minClusterSize = this->minClusterSizeSlot.Param<param::IntParam>()->Value();
    for (auto it = setVec.begin(); it != setVec.end(); /* intentionally empty*/) {
        if ((*it).size() < minClusterSize) {
            it = setVec.erase(it);
        } else {
            ++it;
        }
    }
    std::set<uint64_t> endset;
    for (const auto& s : setVec) {
        endset.insert(s.begin(), s.end());
    }
    return this->copyInCallToContent(call, endset);
}
