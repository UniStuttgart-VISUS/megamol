/*
 * StubModule.cpp
 * Copyright (C) 2017 by MegaMol Team
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/special/StubModule.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/factories/CallAutoDescription.h"

using namespace megamol;
using namespace megamol::core;


special::StubModule::StubModule(void) : Module(),
inSlot("inSlot", "Inbound call"),
outSlot("outSlot", "Outbound call") {

}


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


void special::StubModule::release(void) {

}


bool megamol::core::special::StubModule::stub(Call& c) {
    for (auto cd : this->GetCoreInstance()->GetCallDescriptionManager()) {
        if (cd->IsDescribing(&c)) {
            for (unsigned int idx = 0; idx < cd->FunctionCount(); idx++) {
                try {
                    c(idx);
                } catch (...) {
                    return false;
                }
            }
        }
    }

    return true;
}
