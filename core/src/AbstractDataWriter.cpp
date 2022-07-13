/*
 * AbstractDataWriter.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS)
 * Alle Rechte vorbehalten.
 */

#include "mmcore/AbstractDataWriter.h"
#include "mmcore/DataWriterCtrlCall.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/utility/log/Log.h"

using namespace megamol::core;


/*
 * AbstractDataWriter::AbstractDataWriter
 */
AbstractDataWriter::AbstractDataWriter(void)
        : Module()
        , controlSlot("control", "Slot for incoming control commands")
        , manualRunSlot("manualRun", "Slot fopr manual triggering of the run method.") {

    this->controlSlot.SetCallback(DataWriterCtrlCall::ClassName(),
        DataWriterCtrlCall::FunctionName(DataWriterCtrlCall::CALL_RUN), &AbstractDataWriter::onCallRun);
    this->controlSlot.SetCallback(DataWriterCtrlCall::ClassName(),
        DataWriterCtrlCall::FunctionName(DataWriterCtrlCall::CALL_ABORT), &AbstractDataWriter::onCallAbort);
    this->controlSlot.SetCallback(DataWriterCtrlCall::ClassName(),
        DataWriterCtrlCall::FunctionName(DataWriterCtrlCall::CALL_GETCAPABILITIES),
        &AbstractDataWriter::onCallGetCapability);
    this->MakeSlotAvailable(&this->controlSlot);

    this->manualRunSlot << new core::param::ButtonParam();
    this->manualRunSlot.SetUpdateCallback(&AbstractDataWriter::triggerManualRun);
    this->MakeSlotAvailable(&this->manualRunSlot);
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
bool AbstractDataWriter::onCallRun(Call& call) {
    return this->run();
}


/*
 * AbstractDataWriter::onCallGetCapability
 */
bool AbstractDataWriter::onCallGetCapability(Call& call) {
    DataWriterCtrlCall* dwcc = dynamic_cast<DataWriterCtrlCall*>(&call);
    return (dwcc != NULL) && this->getCapabilities(*dwcc);
}


/*
 * AbstractDataWriter::onCallAbort
 */
bool AbstractDataWriter::onCallAbort(Call& call) {
    return this->abort();
}

/*
 * AbstractDataWriter::triggerManualRun
 */
bool AbstractDataWriter::triggerManualRun(param::ParamSlot& slot) {
    // happy trigger finger hit button action happend
    using megamol::core::utility::log::Log;
    ASSERT(&slot == &this->manualRunSlot);

    Log::DefaultLog.WriteInfo("Manual start initiated ...");

    if (!this->run()) {
        return false;
    }

    return true;
}
