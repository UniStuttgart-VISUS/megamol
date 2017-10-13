/*
 * PBSDataSource.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "PBSDataSource.h"

using namespace megamol;
using namespace megamol::pbs;


PBSDataSource::PBSDataSource(void) : core::Module(),
filename("filename", "The path to the PBS file to load."),
getData("getdata", "Slot to request data from this data source.") {

}


PBSDataSource::~PBSDataSource(void) {
    this->Release();
}


bool PBSDataSource::create(void) {
    return true;
}


void PBSDataSource::release(void) {

}


bool PBSDataSource::filenameChanged(core::param::ParamSlot& slot) {
    return false;
}


bool PBSDataSource::getDataCallback(core::Call& caller) {
    return false;
}


bool PBSDataSource::getExtentCallback(core::Call& caller) {
    return false;
}
