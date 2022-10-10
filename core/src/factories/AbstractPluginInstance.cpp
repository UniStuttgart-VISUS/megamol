/**
 * MegaMol
 * Copyright (c) 2015, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"

using namespace megamol::core;
using namespace megamol::core::factories;

/*
 * AbstractPluginInstance::AbstractPluginInstance
 */
AbstractPluginInstance::AbstractPluginInstance(const char* name, const char* description)
        : call_descriptions()
        , module_descriptions()
        , name(name)
        , description(description)
        , classes_registered(false) {
    if (name == nullptr) {
        throw std::runtime_error("Empty plugin name!");
    }
}

/*
 * AbstractPluginInstance::~AbstractPluginInstance
 */
AbstractPluginInstance::~AbstractPluginInstance() {
    // first remove the descriptions
    this->module_descriptions.Shutdown();
    this->call_descriptions.Shutdown();
}

/*
 * AbstractPluginInstance::GetCallDescriptionManager
 */
const factories::CallDescriptionManager& AbstractPluginInstance::GetCallDescriptionManager() const {
    this->ensureRegisterClassesWrapper();
    return call_descriptions;
}

/*
 * AbstractPluginInstance::GetModuleDescriptionManager
 */
const factories::ModuleDescriptionManager& AbstractPluginInstance::GetModuleDescriptionManager() const {
    this->ensureRegisterClassesWrapper();
    return module_descriptions;
}

/*
 * AbstractPluginInstance::ensureRegisterClassesWrapper
 */
void AbstractPluginInstance::ensureRegisterClassesWrapper() const {
    if (classes_registered) {
        return;
    }
    const_cast<AbstractPluginInstance*>(this)->registerClasses();
    const_cast<AbstractPluginInstance*>(this)->classes_registered = true;
}
