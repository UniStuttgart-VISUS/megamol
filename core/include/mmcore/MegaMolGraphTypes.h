/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

namespace megamol::core {

using ModuleDeletionRequest_t = std::string;

struct ModuleInstantiationRequest {
    std::string className;
    std::string id;
};

using ModuleInstantiationRequest_t = ModuleInstantiationRequest;

struct CallDeletionRequest {
    std::string from;
    std::string to;
};

using CallDeletionRequest_t = CallDeletionRequest;

struct CallInstantiationRequest {
    std::string className;
    std::string from;
    std::string to;
};

using CallInstantiationRequest_t = CallInstantiationRequest;

struct ModuleInstance_t {
    Module::ptr_type modulePtr = nullptr;
    ModuleInstantiationRequest request;
    bool isGraphEntryPoint = false;
    std::vector<std::string> lifetime_resource_requests;
    std::vector<megamol::frontend::FrontendResource> lifetime_resources;
};

using ModuleList_t = std::list<ModuleInstance_t>;

struct CallInstance_t {
    Call::ptr_type callPtr = nullptr;
    CallInstantiationRequest request;
};

using CallList_t = std::list<CallInstance_t>;

} // namespace megamol::core
