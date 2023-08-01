/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd/generic/AbstractStreamProvider.h"
#include "mmcore/Call.h"
#include "mmstd/data/DirectDataWriterCall.h"
#include "mmstd/job/TickCall.h"

#include <functional>

namespace megamol::core {

AbstractStreamProvider::AbstractStreamProvider() : inputSlot("inputSlot", "Slot for providing a callback") {

    this->inputSlot.SetCompatibleCall<DirectDataWriterCall::DirectDataWriterDescription>();
    this->MakeSlotAvailable(&this->inputSlot);
}

AbstractStreamProvider::~AbstractStreamProvider() {
    this->Release();
}

bool AbstractStreamProvider::create() {
    return true;
}

void AbstractStreamProvider::release() {}

bool AbstractStreamProvider::run() {
    auto* call = this->inputSlot.CallAs<DirectDataWriterCall>();

    if (call != nullptr) {
        call->SetCallback(std::bind(&AbstractStreamProvider::GetStream, this));

        return (*call)(0);
    }

    return true;
}

} // namespace megamol::core
