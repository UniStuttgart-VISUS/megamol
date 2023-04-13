/*
 * DataWriterCtrlCall.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS)
 * Alle Rechte vorbehalten.
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
