/**
 * MegaMol
 * Copyright (c) 2017, MegaMol Dev Team
 * All rights reserved.
 */

#include "StubModule.h"

#include "PluginsResource.h"
#include "mmcore/factories/CallAutoDescription.h"

using namespace megamol;
using namespace megamol::core;


special::StubModule::StubModule() : Module(), inSlot("inSlot", "Inbound call"), outSlot("outSlot", "Outbound call") {}


special::StubModule::~StubModule() {
    this->Release();
}


bool special::StubModule::create() {
    auto const& pluginsRes = frontend_resources.get<frontend_resources::PluginsResource>();
    for (auto cd : pluginsRes.all_call_descriptions) {
        this->inSlot.SetCompatibleCall(cd);
        for (unsigned int idx = 0; idx < cd->FunctionCount(); idx++) {
            this->outSlot.SetCallback(cd->ClassName(), cd->FunctionName(idx), &StubModule::stub);
        }
    }

    this->MakeSlotAvailable(&this->inSlot);
    this->MakeSlotAvailable(&this->outSlot);

    return true;
}


void special::StubModule::release() {}


bool megamol::core::special::StubModule::stub(Call& c) {
    auto call = this->inSlot.CallAs<Call>();
    auto const& pluginsRes = frontend_resources.get<frontend_resources::PluginsResource>();
    for (auto cd : pluginsRes.all_call_descriptions) {
        if (cd->IsDescribing(call)) {
            for (unsigned int idx = 0; idx < cd->FunctionCount(); idx++) {
                try {
                    this->inSlot.Call(idx);
                } catch (...) {
                    return false;
                }
            }
        }
    }

    return true;
}
