/*
 * AbstractTickJob.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/job/AbstractTickJob.h"
#include "mmcore/job/TickCall.h"

namespace megamol {
namespace core {
namespace job {

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

} // namespace job
} // namespace core
} // namespace megamol
