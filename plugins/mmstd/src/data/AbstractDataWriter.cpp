/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd/data/AbstractDataWriter.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/utility/log/Log.h"

using namespace megamol::core;


/*
 * AbstractDataWriter::AbstractDataWriter
 */
AbstractDataWriter::AbstractDataWriter()
        : Module()
        , manualRunSlot("manualRun", "Slot fopr manual triggering of the run method.") {

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
 * AbstractDataWriter::triggerManualRun
 */
bool AbstractDataWriter::triggerManualRun(param::ParamSlot& slot) {
    // happy trigger finger hit button action happend
    using megamol::core::utility::log::Log;
    ASSERT(&slot == &this->manualRunSlot);

    Log::DefaultLog.WriteInfo("Manual start initiated ...");

    return this->run();
}
