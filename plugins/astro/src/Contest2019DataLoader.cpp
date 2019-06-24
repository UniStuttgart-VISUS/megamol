/*
 * Contest2019DataLoader.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Contest2019DataLoader.h"
#include <algorithm>
#include <fstream>
#include "astro/AstroDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "vislib/sys/Log.h"

using namespace megamol::core;
using namespace megamol::astro;

#define MAX_MISSED_FILE_NUMBER 5

/*
 * Contest2019DataLoader::Frame::Frame
 */
Contest2019DataLoader::Frame::Frame(view::AnimDataModule& owner) : view::AnimDataModule::Frame(owner) {
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
    if (filepath.empty()) return false;
    this->frame = frameIdx;
    std::vector<SavedData> readDataVec;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        vislib::sys::Log::DefaultLog.WriteError("Could not open input file \"%s\"", filepath.c_str());
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

    this->positions->resize(partCount);
    this->velocities->resize(partCount);
    this->temperatures->resize(partCount);
    this->masses->resize(partCount);
    this->internalEnergies->resize(partCount);
    this->smoothingLengths->resize(partCount);
    this->molecularWeights->resize(partCount);
    this->densities->resize(partCount);
    this->gravitationalPotentials->resize(partCount);
    this->isBaryonFlags->resize(partCount);
    this->isStarFlags->resize(partCount);
    this->isWindFlags->resize(partCount);
    this->isStarFormingGasFlags->resize(partCount);
    this->isAGNFlags->resize(partCount);
    this->particleIDs->resize(partCount);

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
        this->isBaryonFlags->operator[](i) = (s.bitmask >> 1) & 0x1;
        this->isStarFlags->operator[](i) = (s.bitmask >> 5) & 0x1;
        this->isWindFlags->operator[](i) = (s.bitmask >> 6) & 0x1;
        this->isStarFormingGasFlags->operator[](i) = (s.bitmask >> 7) & 0x1;
        this->isAGNFlags->operator[](i) = (s.bitmask >> 8) & 0x1;

        // calculate the temperature ourselves
        // formula out of the mail of J.D Emberson 16.6.2019
        if (this->isBaryonFlags->at(i)) {
            this->temperatures->operator[](i) =
                4.8e5f * this->internalEnergies->at(i) * std::pow(1.0f + redshift, 3.0f);
        }
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
    call.SetTemperature(this->temperatures);
    call.SetMass(this->masses);
    call.SetInternalEnergy(this->internalEnergies);
    call.SetSmoothingLength(this->smoothingLengths);
    call.SetMolecularWeights(this->molecularWeights);
    call.SetDensity(this->densities);
    call.SetGravitationalPotential(this->gravitationalPotentials);
    call.SetIsBaryonFlags(this->isBaryonFlags);
    call.SetIsStarFlags(this->isStarFlags);
    call.SetIsWindFlags(this->isWindFlags);
    call.SetIsStarFormingGasFlags(this->isStarFormingGasFlags);
    call.SetIsAGNFlags(this->isAGNFlags);
    call.SetParticleIDs(this->particleIDs);
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
          "ones from the first given are loaded.") {

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
Contest2019DataLoader::~Contest2019DataLoader(void) { this->Release(); }

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
bool Contest2019DataLoader::create(void) { return true; }

/*
 * Contest2019DataLoader::loadFrame
 */
void Contest2019DataLoader::loadFrame(view::AnimDataModule::Frame* frame, unsigned int idx) {
    using vislib::sys::Log;
    Frame* f = dynamic_cast<Frame*>(frame);
    if (f == nullptr) return;
    unsigned int frameID = idx % this->FrameCount();
    std::string filename = "";
    float redshift = 0.0f;
    if (frameID < this->filenames.size() && frameID < this->redshiftsForFilename.size()) {
        filename = this->filenames.at(frameID);
        redshift = this->redshiftsForFilename.at(frameID);
    }
    if (!filename.empty()) {
        if (!f->LoadFrame(filename, frameID, redshift)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to read frame %d from file\n", idx);
        }
    }
}

/*
 * Contest2019DataLoader::release
 */
void Contest2019DataLoader::release(void) { this->resetFrameCache(); }

/*
 * Contest2019DataLoader::filenameChangedCallback
 */
bool Contest2019DataLoader::filenameChangedCallback(param::ParamSlot& slot) {
    this->filenames.clear();
    this->resetFrameCache();
    this->data_hash++;
    std::string firstfile(this->firstFilename.Param<param::FilePathParam>()->Value());
    int toLoadCount = this->filesToLoad.Param<param::IntParam>()->Value();

    /* Note for all debugging purposes: The application will land here once on startup with only the default values
     * for the input parameters. This first call can be ignored.*/

    if (firstfile.empty()) return false;
    if (toLoadCount == 0) return false;

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
        float redshift = 1.0f / scaleFactor - 1.0f;

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
                vislib::sys::Log::DefaultLog.WriteWarn(
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
            vislib::sys::Log::DefaultLog.WriteWarn(
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
    if (ast == nullptr) return false;

    Frame* f = dynamic_cast<Frame*>(this->requestLockedFrame(ast->FrameID(), ast->IsFrameForced()));
    if (f == nullptr) return false;
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
    if (ast == nullptr) return false;

    ast->SetFrameCount(this->FrameCount());
    ast->AccessBoundingBoxes().Clear();
    ast->AccessBoundingBoxes().SetObjectSpaceBBox(this->boundingBox);
    ast->AccessBoundingBoxes().SetObjectSpaceClipBox(this->clipBox);
    ast->SetDataHash(this->data_hash);

    return true;
}
