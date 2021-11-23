/*
 * DataWriterCtrlCall.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS)
 * Alle Rechte vorbehalten.
 */

#include "mmcore/DataWriterCtrlCall.h"
#include "stdafx.h"

using namespace megamol::core;


/*
 * DataWriterCtrlCall::DataWriterCtrlCall
 */
DataWriterCtrlCall::DataWriterCtrlCall(void) : Call(), abortable(false) {
    // intentionally empty
}


/*
 * DataWriterCtrlCall::~DataWriterCtrlCall
 */
DataWriterCtrlCall::~DataWriterCtrlCall(void) {
    // intentionally empty
}


/*
 * DataWriterCtrlCall::operator=
 */
DataWriterCtrlCall& DataWriterCtrlCall::operator=(const DataWriterCtrlCall& rhs) {
    this->abortable = rhs.abortable;
    return *this;
}
