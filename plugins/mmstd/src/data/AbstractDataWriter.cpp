/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd/data/AbstractDataWriter.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmstd/data/DataWriterCtrlCall.h"

using namespace megamol::core;


/*
 * AbstractDataWriter::AbstractDataWriter
 */
AbstractDataWriter::AbstractDataWriter()
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
AbstractDataWriter::~AbstractDataWriter() {
    // intentionally empty
}


/*
 * AbstractDataWriter::abort
 */
bool AbstractDataWriter::abort() {
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
