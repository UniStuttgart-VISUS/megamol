/*
 * AbstractDataWriter.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractDataWriter.h"
#include "DataWriterCtrlCall.h"

using namespace megamol::core;


/*
 * AbstractDataWriter::AbstractDataWriter
 */
AbstractDataWriter::AbstractDataWriter(void) : Module(),
        controlSlot("control", "Slot for incoming control commands") {

    this->controlSlot.SetCallback(DataWriterCtrlCall::ClassName(),
        DataWriterCtrlCall::FunctionName(DataWriterCtrlCall::CALL_RUN),
        &AbstractDataWriter::onCallRun);
    this->controlSlot.SetCallback(DataWriterCtrlCall::ClassName(),
        DataWriterCtrlCall::FunctionName(DataWriterCtrlCall::CALL_ABORT),
        &AbstractDataWriter::onCallAbort);
    this->controlSlot.SetCallback(DataWriterCtrlCall::ClassName(),
        DataWriterCtrlCall::FunctionName(DataWriterCtrlCall::CALL_GETCAPABILITIES),
        &AbstractDataWriter::onCallGetCapability);
    this->MakeSlotAvailable(&this->controlSlot);
}


/*
 * AbstractDataWriter::~AbstractDataWriter
 */
AbstractDataWriter::~AbstractDataWriter(void) {
    // intentionally empty
}


/*
 * AbstractDataWriter::abort
 */
bool AbstractDataWriter::abort(void) {
    return false; //abort not implemented unless this function is overwritten.
}


/*
 * AbstractDataWriter::onCallRun
 */
bool AbstractDataWriter::onCallRun(Call &call) {
    return this->run();
}


/*
 * AbstractDataWriter::onCallGetCapability
 */
bool AbstractDataWriter::onCallGetCapability(Call &call) {
    DataWriterCtrlCall *dwcc = dynamic_cast<DataWriterCtrlCall*>(&call);
    return (dwcc != NULL) && this->getCapabilities(*dwcc);
}


/*
 * AbstractDataWriter::onCallAbort
 */
bool AbstractDataWriter::onCallAbort(Call &call) {
    return this->abort();
}
