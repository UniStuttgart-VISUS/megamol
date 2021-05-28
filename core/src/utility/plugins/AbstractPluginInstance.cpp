/**
 * MegaMol
 * Copyright (c) 2015-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"

using namespace megamol::core;
using namespace megamol::core::utility::plugins;

/*
 * AbstractPluginInstance::AbstractPluginInstance
 */
AbstractPluginInstance::AbstractPluginInstance(const char* asm_name, const char* description)
        : factories::AbstractAssemblyInstance()
        , asm_name(asm_name)
        , description(description)
        , classes_registered(false) {
    if (asm_name == nullptr) {
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
    return AbstractAssemblyInstance::GetCallDescriptionManager();
}

/*
 * AbstractPluginInstance::GetModuleDescriptionManager
 */
const factories::ModuleDescriptionManager& AbstractPluginInstance::GetModuleDescriptionManager() const {
    this->ensureRegisterClassesWrapper();
    return AbstractAssemblyInstance::GetModuleDescriptionManager();
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
