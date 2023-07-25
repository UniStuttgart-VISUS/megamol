/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "ResourceTestModule.h"


megamol::core::ResourceTestModule::ResourceTestModule() {}


megamol::core::ResourceTestModule::~ResourceTestModule() {
    this->Release();
}


bool megamol::core::ResourceTestModule::create() {
    // we requested this resource, so it is either available or program execution halted before we got here
    auto& cuda_res = frontend_resources.get<frontend_resources::CUDA_Context>();
    if (cuda_res.ctx_ != nullptr) {
        core::utility::log::Log::DefaultLog.WriteInfo("[ResourceTestModule] CUDA context pointer exists");
    }

    return true;
}


void megamol::core::ResourceTestModule::release() {}
