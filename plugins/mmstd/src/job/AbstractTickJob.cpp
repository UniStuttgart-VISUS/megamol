/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd/job/AbstractTickJob.h"
#include "mmstd/job/TickCall.h"

namespace megamol::core::job {

AbstractTickJob::AbstractTickJob() : tickSlot("tickSlot", "Slot for receiving a tick") {
    this->tickSlot.SetCallback(job::TickCall::ClassName(), job::TickCall::FunctionName(0), &AbstractTickJob::Run);
    this->MakeSlotAvailable(&this->tickSlot);
}

AbstractTickJob::~AbstractTickJob() {
    this->Release();
}

bool AbstractTickJob::Run(Call&) {
    this->run();

    return true;
}

} // namespace megamol::core::job
