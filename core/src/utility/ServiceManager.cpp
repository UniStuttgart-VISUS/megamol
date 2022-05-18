/*
 * ServiceManager.cpp
 *
 * Copyright (C) 2016 by MegaMol Team (S. Grottel)
 * Alle Rechte vorbehalten.
 */
#include "utility/ServiceManager.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::utility;

ServiceManager::ServiceManager(CoreInstance& core) : core(core), services() {
    // intentionally empty
}

ServiceManager::~ServiceManager() {
    for (std::shared_ptr<AbstractService> p : services) {
        p->Disable();
        p->Deinitialize();
    }
    services.clear();
}

unsigned int ServiceManager::InstallServiceObject(AbstractService* service, ServiceDeletor deletor) {
    using megamol::core::utility::log::Log;
    if (service == nullptr) {
        Log::DefaultLog.WriteError("Service \"NULL\" cannot be installed.");
        return 0;
    }
    if (deletor == nullptr) {
        Log::DefaultLog.WriteError("Service \"%s\" deletor not set.", service->Name());
        return 0;
    }

    bool enable = false;
    bool succ = service->Initalize(enable);
    if (!succ) {
        Log::DefaultLog.WriteError("Failed to initialize service \"%s\"", service->Name());
        return 0;
    }

    std::shared_ptr<AbstractService> ptr = std::shared_ptr<AbstractService>(service, deletor);
    services.push_back(ptr);
    unsigned int id = static_cast<unsigned int>(services.size());
    Log::DefaultLog.WriteInfo("Installed service \"%s\" [%u]", service->Name(), id);

    if (enable) {
        succ = ptr->Enable();

        if (succ) {
            Log::DefaultLog.WriteInfo("Auto-enabled service \"%s\" [%u]", service->Name(), id);
        } else {
            Log::DefaultLog.WriteError("Failed to auto-enable service \"%s\" [%u]", service->Name(), id);
        }
    }

    return id;
}

AbstractService* ServiceManager::GetInstalledService(unsigned int id) {
    if (id == 0)
        return nullptr;
    if (id > services.size())
        return nullptr;
    return services[id - 1].get();
}
