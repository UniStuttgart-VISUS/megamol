/*
 * DataFileSequencer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "DataFileSequencer.h"
#include "param/ButtonParam.h"
#include "param/StringParam.h"
#include "view/CallRenderView.h"


/*
 * megamol::core::DataFileSequencer::DataFileSequencer
 */
megamol::core::DataFileSequencer::DataFileSequencer(void) : Module(),
        conSlot("connection", "Connects to a view to make this module part of the same module network."),
        filenameSlotNameSlot("filenameSlot", "String parameter identifying the filename slot to manipulate"),
        nextFileSlot("nextFile", "Button parameter to switch to the next file"),
        prevFileSlot("prevFile", "Button parameter to switch to the previous file") {

    this->conSlot.SetCompatibleCall<view::CallRenderViewDescription>();
    // TODO: Connection to a job would also be nice
    this->MakeSlotAvailable(&this->conSlot);

    this->filenameSlotNameSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->filenameSlotNameSlot);

    this->nextFileSlot << new param::ButtonParam('n');
    this->nextFileSlot.SetUpdateCallback(&DataFileSequencer::onNextFile);
    this->MakeSlotAvailable(&this->nextFileSlot);

    this->prevFileSlot << new param::ButtonParam('b');
    this->prevFileSlot.SetUpdateCallback(&DataFileSequencer::onPrevFile);
    this->MakeSlotAvailable(&this->prevFileSlot);
}


/*
 * megamol::core::DataFileSequencer::~DataFileSequencer
 */
megamol::core::DataFileSequencer::~DataFileSequencer(void) {
    this->Release();
    // Intentionally empty
}


/*
 * megamol::core::DataFileSequencer::create
 */
bool megamol::core::DataFileSequencer::create(void) {
    // Intentionally empty
    return true;
}


/*
 * megamol::core::DataFileSequencer::release
 */
void megamol::core::DataFileSequencer::release(void) {
    // Intentionally empty
}


/*
 * megamol::core::DataFileSequencer::onNextFile
 */
bool megamol::core::DataFileSequencer::onNextFile(param::ParamSlot& param) {

    // TODO: Implement

    return true;
}


/*
 * megamol::core::DataFileSequencer::onPrevFile
 */
bool megamol::core::DataFileSequencer::onPrevFile(param::ParamSlot& param) {

    // TODO: Implement

    return true;
}
