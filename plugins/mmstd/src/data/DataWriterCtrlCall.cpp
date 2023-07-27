/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd/data/DataWriterCtrlCall.h"

using namespace megamol::core;


/*
 * DataWriterCtrlCall::DataWriterCtrlCall
 */
DataWriterCtrlCall::DataWriterCtrlCall() : Call(), abortable(false) {
    // intentionally empty
}


/*
 * DataWriterCtrlCall::~DataWriterCtrlCall
 */
DataWriterCtrlCall::~DataWriterCtrlCall() {
    // intentionally empty
}


/*
 * DataWriterCtrlCall::operator=
 */
DataWriterCtrlCall& DataWriterCtrlCall::operator=(const DataWriterCtrlCall& rhs) {
    this->abortable = rhs.abortable;
    return *this;
}
