/*
 * AbstractStreamProvider.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"

#include "mmcore/AbstractStreamProvider.h"
#include "mmcore/Call.h"
#include "mmcore/DirectDataWriterCall.h"
#include "mmcore/job/TickCall.h"

#include <functional>

namespace megamol {
namespace core {

    AbstractStreamProvider::AbstractStreamProvider() :
        inputSlot("inputSlot", "Slot for providing a callback"),
        tickSlot("tickSlot", "Slot for receiving a tick") {

        this->inputSlot.SetCompatibleCall<DirectDataWriterCall::direct_data_writer_description>();
        this->MakeSlotAvailable(&this->inputSlot);

        this->tickSlot.SetCallback(job::TickCall::ClassName(), job::TickCall::FunctionName(0), &AbstractStreamProvider::Run);
        this->MakeSlotAvailable(&this->tickSlot);
    }

    AbstractStreamProvider::~AbstractStreamProvider() {
        this->Release();
    }

    bool AbstractStreamProvider::create() {
        return true;
    }

    void AbstractStreamProvider::release() {
    }

    bool AbstractStreamProvider::Run(Call&) {
        auto* call = this->inputSlot.CallAs<DirectDataWriterCall>();

        if (call != nullptr)
        {
            call->SetCallback(std::bind(&AbstractStreamProvider::GetStream, this));

            return (*call)(0);
        }

        return true;
    }

}
}