/*
 * Contest2019DataLoader.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "Contest2019DataLoader.h"
#include "astro/AstroDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include <algorithm>
#include <fstream>

using namespace megamol::core;
using namespace megamol::astro;

#define MAX_MISSED_FILE_NUMBER 5

/*
 * Contest2019DataLoader::Frame::Frame
 */
Contest2019DataLoader::Frame::Frame(view::AnimDataModule& owner) : view::AnimDataModule::Frame(owner), redshift(0.0f) {
    // intentionally empty
}

/*
 * Contest2019DataLoader::Frame::~Frame
 */
Contest2019DataLoader::Frame::~Frame(void) {
    // all the smart pointers are deleted automatically
}

/*
 * Contest2019DataLoader::Frame::LoadFrame
 */
bool Contest2019DataLoader::Frame::LoadFrame(std::string filepath, unsigned int frameIdx, float redshift) {
    if (filepath.empty())
        return false;
    this->frame = frameIdx;
    std::vector<SavedData> readDataVec;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Could not open input file \"%s\"", filepath.c_str());
        return false;
    }
    // determine size of the file
    file.seekg(0, std::ios_base::end);
    uint64_t size = file.tellg();
    uint64_t partCount = size / sizeof(SavedData);
    readDataVec.resize(partCount);

    // read the data
    file.seekg(0, std::ios_base::beg);
    file.read(reinterpret_cast<char*>(readDataVec.data()), sizeof(SavedData) * partCount);

    // init the fields if necessary
    if (this->positions == nullptr) {
        this->positions = std::make_shared<std::vector<glm::vec3>>();
    }
    if (this->velocities == nullptr) {
        this->velocities = std::make_shared<std::vector<glm::vec3>>();
    }
    if (this->velocityDerivatives == nullptr) {
        this->velocityDerivatives = std::make_shared<std::vector<glm::vec3>>();
    }
    if (this->temperatures == nullptr) {
        this->temperatures = std::make_shared<std::vector<float>>();
    }
    if (this->temperatureDerivatives == nullptr) {
        this->temperatureDerivatives = std::make_shared<std::vector<float>>();
    }
    if (this->masses == nullptr) {
        this->masses = std::make_shared<std::vector<float>>();
    }
    if (this->internalEnergies == nullptr) {
        this->internalEnergies = std::make_shared<std::vector<float>>();
    }
    if (this->internalEnergyDerivatives == nullptr) {
        this->internalEnergyDerivatives = std::make_shared<std::vector<float>>();
    }
    if (this->smoothingLengths == nullptr) {
        this->smoothingLengths = std::make_shared<std::vector<float>>();
    }
    if (this->smoothingLengthDerivatives == nullptr) {
        this->smoothingLengthDerivatives = std::make_shared<std::vector<float>>();
    }
    if (this->molecularWeights == nullptr) {
        this->molecularWeights = std::make_shared<std::vector<float>>();
    }
    if (this->molecularWeightDerivatives == nullptr) {
        this->molecularWeightDerivatives = std::make_shared<std::vector<float>>();
    }
    if (this->densities == nullptr) {
        this->densities = std::make_shared<std::vector<float>>();
    }
    if (this->densityDerivatives == nullptr) {
        this->densityDerivatives = std::make_shared<std::vector<float>>();
    }
    if (this->gravitationalPotentials == nullptr) {
        this->gravitationalPotentials = std::make_shared<std::vector<float>>();
    }
    if (this->gravitationalPotentialDerivatives == nullptr) {
        this->gravitationalPotentialDerivatives = std::make_shared<std::vector<float>>();
    }
    if (this->entropy == nullptr) {
        this->entropy = std::make_shared<std::vector<float>>();
    }
    if (this->entropyDerivatives == nullptr) {
        this->entropyDerivatives = std::make_shared<std::vector<float>>();
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

    this->positions->resize(partCount);
    this->velocities->resize(partCount);
    this->temperatures->resize(partCount);
    this->masses->resize(partCount);
    this->internalEnergies->resize(partCount);
    this->smoothingLengths->resize(partCount);
    this->molecularWeights->resize(partCount);
    this->densities->resize(partCount);
    this->gravitationalPotentials->resize(partCount);
    this->entropy->resize(partCount);
    this->isBaryonFlags->resize(partCount);
    this->isStarFlags->resize(partCount);
    this->isWindFlags->resize(partCount);
    this->isStarFormingGasFlags->resize(partCount);
    this->isAGNFlags->resize(partCount);
    this->particleIDs->resize(partCount);
    this->agnDistances->resize(partCount);

    this->velocityDerivatives->resize(partCount);
    this->temperatureDerivatives->resize(partCount);
    this->internalEnergyDerivatives->resize(partCount);
    this->smoothingLengthDerivatives->resize(partCount);
    this->molecularWeightDerivatives->resize(partCount);
    this->densityDerivatives->resize(partCount);
    this->gravitationalPotentialDerivatives->resize(partCount);
    this->entropyDerivatives->resize(partCount);

    // copy the data over

    this->redshift = redshift;
    for (uint64_t i = 0; i < partCount; ++i) {
        const auto& s = readDataVec[i];
        this->positions->operator[](i) = glm::vec3(s.x, s.y, s.z);
        this->velocities->operator[](i) = glm::vec3(s.vx, s.vy, s.vz);
        this->temperatures->operator[](i) = 0.0f; // oops, we do not have temperatures
        this->masses->operator[](i) = s.mass;
        this->internalEnergies->operator[](i) = s.internalEnergy;
        this->smoothingLengths->operator[](i) = s.smoothingLength;
        this->molecularWeights->operator[](i) = s.molecularWeight;
        this->densities->operator[](i) = s.density;
        this->gravitationalPotentials->operator[](i) = s.gravitationalPotential;
        this->entropy->operator[](i) = 0.0f; // we calculate it later
        this->isBaryonFlags->operator[](i) = (s.bitmask >> 1) & 0x1;
        this->isStarFlags->operator[](i) = (s.bitmask >> 5) & 0x1;
        this->isWindFlags->operator[](i) = (s.bitmask >> 6) & 0x1;
        this->isStarFormingGasFlags->operator[](i) = (s.bitmask >> 7) & 0x1;
        this->isAGNFlags->operator[](i) = (s.bitmask >> 8) & 0x1;
        this->particleIDs->operator[](i) = s.particleID;

        // calculate the temperature ourselves
        // formula out of the mail of J.D Emberson 20.6.2019
        if (this->isBaryonFlags->at(i)) {
            this->temperatures->operator[](i) =
                4.8e5f * this->internalEnergies->at(i) / std::pow(1.0f + redshift, 3.0f);
        }

        // calculate the entropy ourselves
        // formula directly from the contest description
        if (this->isBaryonFlags->at(i) && this->temperatures->at(i) > 0.0f && this->densities->at(i) > 0.0f) {
            auto t = (*this->temperatures)[i];
            auto p = (*this->densities)[i];
            this->entropy->operator[](i) = std::log(t / std::pow(p, 2.0f / 3.0f));

            // This is Juhans formula:
            // auto mu = (*this->masses)[i];
            // auto eps = (*this->internalEnergies)[i];
            //(*this->entropy)[i] = std::log((mu * eps) / std::pow(p, 2.0f / 3.0f));
        }

        // the derivatives will be calculated later, when the frame before and after are known
    }
    return true;
}

/*
 * Contest2019DataLoader::Frame::SetData
 */
void Contest2019DataLoader::Frame::SetData(
    AstroDataCall& call, const vislib::math::Cuboid<float>& boundingBox, const vislib::math::Cuboid<float>& clipBox) {
    if (this->positions == nullptr || this->positions->empty()) {
        call.ClearValues();
    }
    call.SetPositions(this->positions);
    call.SetVelocities(this->velocities);
    call.SetVelocityDerivatives(this->velocityDerivatives);
    call.SetTemperature(this->temperatures);
    call.SetTemperatureDerivatives(this->temperatureDerivatives);
    call.SetMass(this->masses);
    call.SetInternalEnergy(this->internalEnergies);
    call.SetInternalEnergyDerivatives(this->internalEnergyDerivatives);
    call.SetSmoothingLength(this->smoothingLengths);
    call.SetSmoothingLengthDerivatives(this->smoothingLengthDerivatives);
    call.SetMolecularWeights(this->molecularWeights);
    call.SetMolecularWeightDerivatives(this->molecularWeightDerivatives);
    call.SetDensity(this->densities);
    call.SetDensityDerivative(this->densityDerivatives);
    call.SetGravitationalPotential(this->gravitationalPotentials);
    call.SetGravitationalPotentialDerivatives(this->gravitationalPotentialDerivatives);
    call.SetEntropy(this->entropy);
    call.SetEntropyDerivatives(this->entropyDerivatives);
    call.SetIsBaryonFlags(this->isBaryonFlags);
    call.SetIsStarFlags(this->isStarFlags);
    call.SetIsWindFlags(this->isWindFlags);
    call.SetIsStarFormingGasFlags(this->isStarFormingGasFlags);
    call.SetIsAGNFlags(this->isAGNFlags);
    call.SetParticleIDs(this->particleIDs);
    call.SetAGNDistances(this->agnDistances);
}

/*
 * Contest2019DataLoader::Frame::ZeroDerivatives
 */
void Contest2019DataLoader::Frame::ZeroDerivatives(void) {
    if (this->velocityDerivatives != nullptr) {
        std::fill(this->velocityDerivatives->begin(), this->velocityDerivatives->end(), glm::vec3(0.0f));
    }
    if (this->temperatureDerivatives != nullptr) {
        std::fill(this->temperatureDerivatives->begin(), this->temperatureDerivatives->end(), 0.0f);
    }
    if (this->internalEnergyDerivatives != nullptr) {
        std::fill(this->internalEnergyDerivatives->begin(), this->internalEnergyDerivatives->end(), 0.0f);
    }
    if (this->smoothingLengthDerivatives != nullptr) {
        std::fill(this->smoothingLengthDerivatives->begin(), this->smoothingLengthDerivatives->end(), 0.0f);
    }
    if (this->molecularWeightDerivatives != nullptr) {
        std::fill(this->molecularWeightDerivatives->begin(), this->molecularWeightDerivatives->end(), 0.0f);
    }
    if (this->densityDerivatives != nullptr) {
        std::fill(this->densityDerivatives->begin(), this->densityDerivatives->end(), 0.0f);
    }
    if (this->gravitationalPotentialDerivatives != nullptr) {
        std::fill(
            this->gravitationalPotentialDerivatives->begin(), this->gravitationalPotentialDerivatives->end(), 0.0f);
    }
    if (this->entropyDerivatives != nullptr) {
        std::fill(this->entropyDerivatives->begin(), this->entropyDerivatives->end(), 0.0f);
    }
}

/*
 * Contest2019DataLoader::Frame::ZeroAGNDistances
 */
void Contest2019DataLoader::Frame::ZeroAGNDistances(void) {
    if (this->agnDistances != nullptr) {
        std::fill(this->agnDistances->begin(), this->agnDistances->end(), 0.0f);
    }
}

/*
 * Contest2019DataLoader::Frame::CalculateDerivatives
 */
void Contest2019DataLoader::Frame::CalculateDerivatives(
    Contest2019DataLoader::Frame* frameBefore, Contest2019DataLoader::Frame* frameAfter) {
    // if there is only one frame we leave the derivatives at 0
    if ((frameBefore->frame == frameAfter->frame) && (this->frame == frameBefore->frame)) {
        return;
    }
    if (frameBefore->frame == this->frame) {
        this->CalculateDerivativesForwardDifferences(frameAfter);
        return;
    }
    if (frameAfter->frame == this->frame) {
        this->CalculateDerivativesBackwardDifferences(frameAfter);
        return;
    }
    this->CalculateDerivativesCentralDifferences(frameBefore, frameAfter);
}

/*
 * Contest2019DataLoader::Frame::buildParticleIDMap
 */
void Contest2019DataLoader::Frame::buildParticleIDMap(const Frame* frame, std::map<int64_t, int64_t>& outIndexMap) {
    outIndexMap.clear();
    if (frame != nullptr && frame->particleIDs != nullptr) {
        for (int64_t i = 0; i < frame->particleIDs->size(); i++) {
            outIndexMap.insert(std::pair<int64_t, int64_t>(frame->particleIDs->at(i), i));
        }
    }
}

/*
 * Contest2019DataLoader::Frame::CalculateDerivativesBackwardDifferences
 */
void Contest2019DataLoader::Frame::CalculateDerivativesBackwardDifferences(Contest2019DataLoader::Frame* frameBefore) {
    if (this->particleIDs == nullptr)
        return;
    std::map<int64_t, int64_t> mapBefore;
    this->buildParticleIDMap(frameBefore, mapBefore);
    int64_t idbefore;
    for (int64_t i = 0; i < this->particleIDs->size(); ++i) {
        idbefore = (mapBefore.count(this->particleIDs->at(i)) > 0) ? mapBefore[this->particleIDs->at(i)] : -1;
        if (idbefore >= 0) {
            this->velocityDerivatives->at(i) =
                backwardDifference(this->velocities->at(i), frameBefore->velocities->at(idbefore));
            this->temperatureDerivatives->at(i) =
                backwardDifference(this->temperatures->at(i), frameBefore->temperatures->at(idbefore));
            this->internalEnergyDerivatives->at(i) =
                backwardDifference(this->internalEnergies->at(i), frameBefore->internalEnergies->at(idbefore));
            this->smoothingLengthDerivatives->at(i) =
                backwardDifference(this->smoothingLengths->at(i), frameBefore->smoothingLengths->at(idbefore));
            this->molecularWeightDerivatives->at(i) =
                backwardDifference(this->molecularWeights->at(i), frameBefore->molecularWeights->at(idbefore));
            this->densityDerivatives->at(i) =
                backwardDifference(this->densities->at(i), frameBefore->densities->at(idbefore));
            this->gravitationalPotentialDerivatives->at(i) = backwardDifference(
                this->gravitationalPotentials->at(i), frameBefore->gravitationalPotentials->at(idbefore));
            this->entropyDerivatives->at(i) =
                backwardDifference(this->entropy->at(i), frameBefore->entropy->at(idbefore));
        }
    }
}

/*
 * Contest2019DataLoader::Frame::CalculateDerivativesForwardDifferences
 */
void Contest2019DataLoader::Frame::CalculateDerivativesForwardDifferences(Contest2019DataLoader::Frame* frameAfter) {
    if (this->particleIDs == nullptr)
        return;
    std::map<int64_t, int64_t> mapAfter;
    this->buildParticleIDMap(frameAfter, mapAfter);
    int64_t idafter;
    for (int64_t i = 0; i < this->particleIDs->size(); ++i) {
        idafter = (mapAfter.count(this->particleIDs->at(i)) > 0) ? mapAfter[this->particleIDs->at(i)] : -1;
        if (idafter >= 0) {
            this->velocityDerivatives->at(i) =
                forwardDifference(this->velocities->at(i), frameAfter->velocities->at(idafter));
            this->temperatureDerivatives->at(i) =
                forwardDifference(this->temperatures->at(i), frameAfter->temperatures->at(idafter));
            this->internalEnergyDerivatives->at(i) =
                forwardDifference(this->internalEnergies->at(i), frameAfter->internalEnergies->at(idafter));
            this->smoothingLengthDerivatives->at(i) =
                forwardDifference(this->smoothingLengths->at(i), frameAfter->smoothingLengths->at(idafter));
            this->molecularWeightDerivatives->at(i) =
                forwardDifference(this->molecularWeights->at(i), frameAfter->molecularWeights->at(idafter));
            this->densityDerivatives->at(i) =
                forwardDifference(this->densities->at(i), frameAfter->densities->at(idafter));
            this->gravitationalPotentialDerivatives->at(i) = forwardDifference(
                this->gravitationalPotentials->at(i), frameAfter->gravitationalPotentials->at(idafter));
            this->entropyDerivatives->at(i) = forwardDifference(this->entropy->at(i), frameAfter->entropy->at(idafter));
        }
    }
}

/*
 * Contest2019DataLoader::Frame::CalculateDerivativesCentralDifferences
 */
void Contest2019DataLoader::Frame::CalculateDerivativesCentralDifferences(
    Contest2019DataLoader::Frame* frameBefore, Contest2019DataLoader::Frame* frameAfter) {
    if (this->particleIDs == nullptr)
        return;
    std::map<int64_t, int64_t> mapBefore, mapAfter;
    this->buildParticleIDMap(frameBefore, mapBefore);
    this->buildParticleIDMap(frameAfter, mapAfter);
    int64_t idbefore, idafter;
    for (int64_t i = 0; i < this->particleIDs->size(); ++i) {
        // retrieve indices in other frames
        idbefore = (mapBefore.count(this->particleIDs->at(i)) > 0) ? mapBefore[this->particleIDs->at(i)] : -1;
        idafter = (mapAfter.count(this->particleIDs->at(i)) > 0) ? mapAfter[this->particleIDs->at(i)] : -1;
        // fallback to other difference modes if some particle ids are not available
        if (idbefore >= 0 && idafter >= 0) {
            this->velocityDerivatives->at(i) =
                centralDifference(frameBefore->velocities->at(idbefore), frameAfter->velocities->at(idafter));
            this->temperatureDerivatives->at(i) =
                centralDifference(frameBefore->temperatures->at(idbefore), frameAfter->temperatures->at(idafter));
            this->internalEnergyDerivatives->at(i) = centralDifference(
                frameBefore->internalEnergies->at(idbefore), frameAfter->internalEnergies->at(idafter));
            this->smoothingLengthDerivatives->at(i) = centralDifference(
                frameBefore->smoothingLengths->at(idbefore), frameAfter->smoothingLengths->at(idafter));
            this->molecularWeightDerivatives->at(i) = centralDifference(
                frameBefore->molecularWeights->at(idbefore), frameAfter->molecularWeights->at(idafter));
            this->densityDerivatives->at(i) =
                centralDifference(frameBefore->densities->at(idbefore), frameAfter->densities->at(idafter));
            this->gravitationalPotentialDerivatives->at(i) = centralDifference(
                frameBefore->gravitationalPotentials->at(idbefore), frameAfter->gravitationalPotentials->at(idafter));
            this->entropyDerivatives->at(i) =
                centralDifference(frameBefore->entropy->at(idbefore), frameAfter->entropy->at(idafter));
        } else if (idbefore < 0 && idafter >= 0) {
            this->velocityDerivatives->at(i) =
                forwardDifference(this->velocities->at(i), frameAfter->velocities->at(idafter));
            this->temperatureDerivatives->at(i) =
                forwardDifference(this->temperatures->at(i), frameAfter->temperatures->at(idafter));
            this->internalEnergyDerivatives->at(i) =
                forwardDifference(this->internalEnergies->at(i), frameAfter->internalEnergies->at(idafter));
            this->smoothingLengthDerivatives->at(i) =
                forwardDifference(this->smoothingLengths->at(i), frameAfter->smoothingLengths->at(idafter));
            this->molecularWeightDerivatives->at(i) =
                forwardDifference(this->molecularWeights->at(i), frameAfter->molecularWeights->at(idafter));
            this->densityDerivatives->at(i) =
                forwardDifference(this->densities->at(i), frameAfter->densities->at(idafter));
            this->gravitationalPotentialDerivatives->at(i) = forwardDifference(
                this->gravitationalPotentials->at(i), frameAfter->gravitationalPotentials->at(idafter));
            this->entropyDerivatives->at(i) = forwardDifference(this->entropy->at(i), frameAfter->entropy->at(idafter));
        } else if (idafter < 0 && idbefore >= 0) {
            this->velocityDerivatives->at(i) =
                backwardDifference(this->velocities->at(i), frameBefore->velocities->at(idbefore));
            this->temperatureDerivatives->at(i) =
                backwardDifference(this->temperatures->at(i), frameBefore->temperatures->at(idbefore));
            this->internalEnergyDerivatives->at(i) =
                backwardDifference(this->internalEnergies->at(i), frameBefore->internalEnergies->at(idbefore));
            this->smoothingLengthDerivatives->at(i) =
                backwardDifference(this->smoothingLengths->at(i), frameBefore->smoothingLengths->at(idbefore));
            this->molecularWeightDerivatives->at(i) =
                backwardDifference(this->molecularWeights->at(i), frameBefore->molecularWeights->at(idbefore));
            this->densityDerivatives->at(i) =
                backwardDifference(this->densities->at(i), frameBefore->densities->at(idbefore));
            this->gravitationalPotentialDerivatives->at(i) = backwardDifference(
                this->gravitationalPotentials->at(i), frameBefore->gravitationalPotentials->at(idbefore));
            this->entropyDerivatives->at(i) =
                backwardDifference(this->entropy->at(i), frameBefore->entropy->at(idbefore));
        }
    }
}

/*
 * Contest2019DataLoader::Frame::CalculateAGNDistances
 */
void Contest2019DataLoader::Frame::CalculateAGNDistances(void) {
    if (this->positions == nullptr)
        return;
    if (this->isAGNFlags == nullptr)
        return;
    if (this->agnDistances == nullptr)
        return;

    // get out all AGN Positions
    std::vector<glm::vec3> agnPositions;
    for (size_t i = 0; i < this->positions->size(); ++i) {
        if (this->isAGNFlags->at(i)) {
            agnPositions.push_back(this->positions->at(i));
        }
    }
    // add all mirrored version to account for cyclic boundary conditions
    std::vector<glm::vec3> apos;
    for (size_t i = 0; i < agnPositions.size(); i++) {
        const auto& pos = agnPositions.at(i);
        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                for (int z = -1; z <= 1; z++) {
                    apos.push_back(pos + glm::vec3(static_cast<float>(x * 64.0f), static_cast<float>(y * 64.0f),
                                             static_cast<float>(z * 64.0f)));
                }
            }
        }
    }

    if (apos.size() == 0)
        return;
    for (size_t i = 0; i < this->positions->size(); ++i) {
        float mindist = std::numeric_limits<float>::max();
        auto& myPos = this->positions->at(i);
        for (const auto& agnPos : apos) {
            float dist = glm::distance(myPos, agnPos);
            if (dist < mindist)
                mindist = dist;
        }
        this->agnDistances->at(i) = mindist;
    }
}

/*
 * Contest2019DataLoader::Contest2019DataLoader
 */
Contest2019DataLoader::Contest2019DataLoader(void)
        : view::AnimDataModule()
        , getDataSlot("getData", "Slot for handling the file loading requests")
        , firstFilename("firstFilename", "The name of the first file to load")
        , filesToLoad("filesToLoad",
              "The total number of files that should be loaded. A value smaller than 0 means all available "
              "ones from the first given are loaded.")
        , calculateDerivatives("calculateDerivatives",
              "Enables the calculation of derivatives of all relevant values. "
              "This option increases the frame loading time significantly. The effect of this slot might be delayed as "
              "already existing frames are not re-evaluated.")
        , calculateAGNDistances("calculateAGNDistances",
              "Enables the calculation of the distance to the AGNs. This option increases the frame loading time "
              "significantly. The effect of this slot might be delayed as already existing frames are not "
              "re-evaluated.") {

    this->getDataSlot.SetCallback(AstroDataCall::ClassName(),
        AstroDataCall::FunctionName(AstroDataCall::CallForGetData), &Contest2019DataLoader::getDataCallback);
    this->getDataSlot.SetCallback(AstroDataCall::ClassName(),
        AstroDataCall::FunctionName(AstroDataCall::CallForGetExtent), &Contest2019DataLoader::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

    this->firstFilename.SetParameter(new param::FilePathParam(""));
    this->firstFilename.SetUpdateCallback(&Contest2019DataLoader::filenameChangedCallback);
    this->MakeSlotAvailable(&this->firstFilename);

    this->filesToLoad.SetParameter(new param::IntParam(-1));
    this->filesToLoad.SetUpdateCallback(&Contest2019DataLoader::filenameChangedCallback);
    this->MakeSlotAvailable(&this->filesToLoad);

    this->calculateDerivatives.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->calculateDerivatives);

    this->calculateAGNDistances.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->calculateAGNDistances);

    // static bounding box size, because we know (TM)
    this->boundingBox = vislib::math::Cuboid<float>(0.0f, 0.0f, 0.0f, 64.0f, 64.0f, 64.0f);
    this->clipBox = this->boundingBox;

    this->data_hash = 0;

    this->setFrameCount(1);
    this->initFrameCache(1);
}

/*
 * Contest2019DataLoader::~Contest2019DataLoader
 */
Contest2019DataLoader::~Contest2019DataLoader(void) {
    this->Release();
}

/*
 * Contest2019DataLoader::constructFrame
 */
view::AnimDataModule::Frame* Contest2019DataLoader::constructFrame(void) const {
    Frame* f = new Frame(*const_cast<Contest2019DataLoader*>(this));
    return f;
}

/*
 * Contest2019DataLoader::create
 */
bool Contest2019DataLoader::create(void) {
    return true;
}

/*
 * Contest2019DataLoader::loadFrame
 */
void Contest2019DataLoader::loadFrame(view::AnimDataModule::Frame* frame, unsigned int idx) {
    using megamol::core::utility::log::Log;
    Frame* f = dynamic_cast<Frame*>(frame);
    // the allocation of the dummy frames here is stupid and should be done globally to avoid too many allocations.
    // the parallel nature of the AnimDataModule makes this impossible
    Frame* fbefore = new Frame(*this);
    Frame* fafter = new Frame(*this);
    if (f == nullptr)
        return;
    unsigned int frameID = idx % this->FrameCount();
    unsigned int frameIDBefore = frameID > 0 ? (frameID - 1) : frameID;
    unsigned int frameIDAfter = frameID < this->redshiftsForFilename.size() - 1 ? (frameID + 1) : frameID;
    std::string filename = "";
    std::string filenameBefore = "";
    std::string filenameAfter = "";
    float redshift = 0.0f;
    float redshiftBefore = 0.0f;
    float redshiftAfter = 0.0f;
    if (frameID < this->filenames.size() && frameID < this->redshiftsForFilename.size()) {
        filename = this->filenames.at(frameID);
        redshift = this->redshiftsForFilename.at(frameID);
    }
    if (frameIDBefore < this->filenames.size() && frameIDBefore < this->redshiftsForFilename.size()) {
        filenameBefore = this->filenames.at(frameIDBefore);
        redshiftBefore = this->redshiftsForFilename.at(frameIDBefore);
    }
    if (frameIDAfter < this->filenames.size() && frameIDAfter < this->redshiftsForFilename.size()) {
        filenameAfter = this->filenames.at(frameIDAfter);
        redshiftAfter = this->redshiftsForFilename.at(frameIDAfter);
    }
    if (!filename.empty()) {
        if (!f->LoadFrame(filename, frameID, redshift)) {
            Log::DefaultLog.WriteError("Unable to read frame %d from file\n", idx);
        }
    }
    bool calcDerivatives = this->calculateDerivatives.Param<param::BoolParam>()->Value();
    if (!filenameBefore.empty() && calcDerivatives) {
        if (!fbefore->LoadFrame(filenameBefore, frameIDBefore, redshiftBefore)) {
            Log::DefaultLog.WriteError("Unable to read frame before frame %d from file\n", idx);
        }
    }
    if (!filenameAfter.empty() && calcDerivatives) {
        if (!fafter->LoadFrame(filenameAfter, frameIDAfter, redshiftAfter)) {
            Log::DefaultLog.WriteError("Unable to read frame after frame %d from file\n", idx);
        }
    }
    f->ZeroDerivatives();
    if (calcDerivatives) {
        f->CalculateDerivatives(fbefore, fafter);
    }
    f->ZeroAGNDistances();
    if (this->calculateAGNDistances.Param<param::BoolParam>()->Value()) {
        f->CalculateAGNDistances();
    }
    delete fbefore;
    delete fafter;
}

/*
 * Contest2019DataLoader::release
 */
void Contest2019DataLoader::release(void) {
    this->resetFrameCache();
}

/*
 * Contest2019DataLoader::filenameChangedCallback
 */
bool Contest2019DataLoader::filenameChangedCallback(param::ParamSlot& slot) {
    this->filenames.clear();
    this->resetFrameCache();
    this->data_hash++;
    std::string firstfile(this->firstFilename.Param<param::FilePathParam>()->Value().generic_u8string());
    int toLoadCount = this->filesToLoad.Param<param::IntParam>()->Value();

    /* Note for all debugging purposes: The application will land here once on startup with only the default values
     * for the input parameters. This first call can be ignored.*/

    if (firstfile.empty())
        return false;
    if (toLoadCount == 0)
        return false;

    auto lastPoint = firstfile.find_last_of('.');
    std::string prefix = firstfile.substr(0, lastPoint + 1);
    std::string postfix = firstfile.substr(lastPoint + 1);
    int firstID = std::stoi(postfix);
    int curID = firstID;
    int loadedCounter = 0;
    int missedFiles = 0;

    bool done = false;
    while (!done) {
        std::string curFilename = prefix + std::to_string(curID);

        // TODO: WARNING: This scalefactor calculation is dependant on the used data set containing 625 time steps
        // starting at z=200 going to z=0. For other data set sizes this calculation has to be adapted. (The physicists
        // were too stupid to include this value into the data)
        float scaleFactor = 1.0f / 201.0f + static_cast<float>(curID + 1) * (1.0f - 1.0f / 201.0f) / 625.0f;
        float redshift = (1.0f / scaleFactor) - 1.0f;

        std::ifstream file(curFilename);
        if (file.good()) {
            file.close();
            this->filenames.push_back(curFilename);
            this->redshiftsForFilename.push_back(redshift);
        } else {
            file.close();
            if (toLoadCount < 0) {
                done = true;
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "Could not find the suggested input file \"%s\"", curFilename.c_str());
                missedFiles++;
            }
        }
        loadedCounter++;
        curID++;
        if (loadedCounter >= toLoadCount && toLoadCount > 0) {
            done = true;
        }
        if (missedFiles > MAX_MISSED_FILE_NUMBER) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "Already could not open %i files, aborting further checking", int(MAX_MISSED_FILE_NUMBER));
            done = true;
            return false;
        }
    }
    this->setFrameCount(static_cast<unsigned int>(this->filenames.size()));
    this->initFrameCache(std::min(this->FrameCount(), 100u)); // TODO change this to a dynamic / user selected value

    return true;
}

/*
 * Contest2019DataLoader::getDataCallback
 */
bool Contest2019DataLoader::getDataCallback(Call& caller) {
    AstroDataCall* ast = dynamic_cast<AstroDataCall*>(&caller);
    if (ast == nullptr)
        return false;

    Frame* f = dynamic_cast<Frame*>(this->requestLockedFrame(ast->FrameID(), ast->IsFrameForced()));
    if (f == nullptr)
        return false;
    ast->SetUnlocker(new Unlocker(*f));
    ast->SetFrameID(f->FrameNumber());
    ast->SetDataHash(this->data_hash);
    f->SetData(*ast, this->boundingBox, this->clipBox);

    return true;
}

/*
 * Contest2019DataLoader::getExtentCallback
 */
bool Contest2019DataLoader::getExtentCallback(Call& caller) {
    AstroDataCall* ast = dynamic_cast<AstroDataCall*>(&caller);
    if (ast == nullptr)
        return false;

    ast->SetFrameCount(this->FrameCount());
    ast->AccessBoundingBoxes().Clear();
    ast->AccessBoundingBoxes().SetObjectSpaceBBox(this->boundingBox);
    ast->AccessBoundingBoxes().SetObjectSpaceClipBox(this->clipBox);
    ast->SetDataHash(this->data_hash);

    return true;
}
