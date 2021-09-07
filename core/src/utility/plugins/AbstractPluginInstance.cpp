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
AbstractPluginInstance::AbstractPluginInstance(const char* name, const char* description)
        : factories::AbstractObjectFactoryInstance()
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
    return AbstractObjectFactoryInstance::GetCallDescriptionManager();
}

/*
 * AbstractPluginInstance::GetModuleDescriptionManager
 */
const factories::ModuleDescriptionManager& AbstractPluginInstance::GetModuleDescriptionManager() const {
    this->ensureRegisterClassesWrapper();
    return AbstractObjectFactoryInstance::GetModuleDescriptionManager();
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
