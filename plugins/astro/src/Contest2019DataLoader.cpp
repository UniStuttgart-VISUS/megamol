/*
 * Contest2019DataLoader.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Contest2019DataLoader.h"
#include "astro/AstroDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "vislib/sys/Log.h"

using namespace megamol::core;
using namespace megamol::astro;

Contest2019DataLoader::Frame::Frame(view::AnimDataModule& owner) : view::AnimDataModule::Frame(owner) {
    // intentionally empty
}

Contest2019DataLoader::Frame::~Frame(void) {
    // all the smart pointers are deleted automatically
}

// TODO Frame::loadFrame
bool Contest2019DataLoader::Frame::LoadFrame(std::string filepath, unsigned int frameIdx) { return true; }

void Contest2019DataLoader::Frame::SetData(AstroDataCall& call) {}

// TODO Frame::setData

Contest2019DataLoader::Contest2019DataLoader(void)
    : view::AnimDataModule()
    , getDataSlot("getData", "Slot for handling the file loading requests")
    , firstFilename("firstFilename", "The name of the first file to load")
    , filesToLoad("filesToLoad", "The total number of files that should be loaded. A value smaller than 0 means all available "
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
    this->MakeSlotAvailable(&this->filesToLoad);

    this->setFrameCount(1);
    this->initFrameCache(1);
}

Contest2019DataLoader::~Contest2019DataLoader(void) { this->Release(); }

view::AnimDataModule::Frame* Contest2019DataLoader::constructFrame(void) const {
    Frame* f = new Frame(*const_cast<Contest2019DataLoader*>(this));
    return f;
}

bool Contest2019DataLoader::create(void) { return true; }

void Contest2019DataLoader::loadFrame(view::AnimDataModule::Frame* frame, unsigned int idx) {
    using vislib::sys::Log;
    Frame* f = dynamic_cast<Frame*>(frame);
    if (f == nullptr) return;
    ASSERT(idx < this->FrameCount());
    std::string filename; // TODO determine correct filename and frame idx
    unsigned int frameID;
    if (!f->LoadFrame(filename, frameID)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to read frame %d from file\n", idx);
    }
}

void Contest2019DataLoader::release(void) { this->resetFrameCache(); }

bool Contest2019DataLoader::filenameChangedCallback(param::ParamSlot& slot) {
    // TODO implement
    return true;
}

bool Contest2019DataLoader::getDataCallback(Call& caller) {
    AstroDataCall* ast = dynamic_cast<AstroDataCall*>(&caller);
    if (ast == nullptr) return false;

    Frame* f = dynamic_cast<Frame*>(this->requestLockedFrame(ast->FrameID(), ast->IsFrameForced()));
    if (f == nullptr) return false;
    ast->SetUnlocker(new Unlocker(*f));
    ast->SetFrameID(f->FrameNumber());
    ast->SetDataHash(this->data_hash);
    f->SetData(*ast);

    return true;
}

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
