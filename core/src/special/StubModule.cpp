/*
 * StubModule.cpp
 * Copyright (C) 2017 by MegaMol Team
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "mmcore/special/StubModule.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/factories/CallAutoDescription.h"

using namespace megamol;
using namespace megamol::core;


special::StubModule::StubModule(void)
        : Module()
        , inSlot("inSlot", "Inbound call")
        , outSlot("outSlot", "Outbound call") {}


special::StubModule::~StubModule(void) {
    this->Release();
}


bool special::StubModule::create(void) {
    for (auto cd : this->GetCoreInstance()->GetCallDescriptionManager()) {
        this->inSlot.SetCompatibleCall(cd);
        for (unsigned int idx = 0; idx < cd->FunctionCount(); idx++) {
            this->outSlot.SetCallback(cd->ClassName(), cd->FunctionName(idx), &StubModule::stub);
        }
    }

    this->MakeSlotAvailable(&this->inSlot);
    this->MakeSlotAvailable(&this->outSlot);

    return true;
}


void special::StubModule::release(void) {}


bool megamol::core::special::StubModule::stub(Call& c) {
    auto call = this->inSlot.CallAs<Call>();
    for (auto cd : this->GetCoreInstance()->GetCallDescriptionManager()) {
        if (cd->IsDescribing(call)) {
            for (unsigned int idx = 0; idx < cd->FunctionCount(); idx++) {
                try {
                    this->inSlot.Call(idx);
                } catch (...) { return false; }
            }
        }
    }

    return true;
}
