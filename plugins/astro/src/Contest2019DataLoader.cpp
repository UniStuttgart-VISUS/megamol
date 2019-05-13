/*
 * Contest2019DataLoader.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Contest2019DataLoader.h"
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
bool Contest2019DataLoader::Frame::LoadFrame(std::string filepath, unsigned int frameIdx) {
    this->frame = frameIdx;
    // TODO implement
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
    ASSERT(idx < this->FrameCount());
    unsigned int frameID = idx % this->FrameCount();
    std::string filename = "";
    if (frameID < this->filenames.size()) {
        filename = this->filenames.at(frameID);
    }
    if (!f->LoadFrame(filename, frameID)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to read frame %d from file\n", idx);
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
    std::string firstfile = T2A(this->firstFilename.Param<param::FilePathParam>()->Value());
    int toLoadCount = this->filesToLoad.Param<param::IntParam>()->Value();

    /* Note for all debugging purposes: The application will land here once on startup with only the default values for
     * the input parameters. This first call can be ignored.*/

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
        std::ifstream file(curFilename);
        if (file.good()) {
            file.close();
            this->filenames.push_back(curFilename);
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
