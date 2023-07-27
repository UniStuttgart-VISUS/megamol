/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "TickSwitch.h"

#include "mmcore/Call.h"
#include "mmstd/job/TickCall.h"

namespace megamol::core::job {

TickSwitch::TickSwitch()
        : incoming_slot("tick_slot", "Tick")
        , outgoing_slot_1("out_tick_slot_1", "Tick")
        , outgoing_slot_2("out_tick_slot_2", "Tick")
        , outgoing_slot_3("out_tick_slot_3", "Tick")
        , outgoing_slot_4("out_tick_slot_4", "Tick") {

    this->incoming_slot.SetCallback(TickCall::ClassName(), TickCall::FunctionName(0), &TickSwitch::TickCallback);
    this->MakeSlotAvailable(&this->incoming_slot);

    this->outgoing_slot_1.SetCompatibleCall<TickCall::TickCallDescription>();
    this->MakeSlotAvailable(&this->outgoing_slot_1);

    this->outgoing_slot_2.SetCompatibleCall<TickCall::TickCallDescription>();
    this->MakeSlotAvailable(&this->outgoing_slot_2);

    this->outgoing_slot_3.SetCompatibleCall<TickCall::TickCallDescription>();
    this->MakeSlotAvailable(&this->outgoing_slot_3);

    this->outgoing_slot_4.SetCompatibleCall<TickCall::TickCallDescription>();
    this->MakeSlotAvailable(&this->outgoing_slot_4);
}

TickSwitch::~TickSwitch() {
    this->Release();
}

bool TickSwitch::TickCallback(core::Call& call) {

    auto* outgoing_call = outgoing_slot_1.CallAs<TickCall>();

    if (outgoing_call != nullptr) {
        (*outgoing_call)(0);
    }

    outgoing_call = outgoing_slot_2.CallAs<TickCall>();

    if (outgoing_call != nullptr) {
        (*outgoing_call)(0);
    }

    outgoing_call = outgoing_slot_3.CallAs<TickCall>();

    if (outgoing_call != nullptr) {
        (*outgoing_call)(0);
    }

    outgoing_call = outgoing_slot_4.CallAs<TickCall>();

    if (outgoing_call != nullptr) {
        (*outgoing_call)(0);
    }

    return true;
}

} // namespace megamol::core::job
