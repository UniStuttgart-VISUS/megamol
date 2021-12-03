/**
 * MegaMol
 * Copyright (c) 2008-2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOLCORE_FACTORIES_MODULEDESCRIPTIONMANAGER_H_INCLUDED
#define MEGAMOLCORE_FACTORIES_MODULEDESCRIPTIONMANAGER_H_INCLUDED
#pragma once

#include "ModuleAutoDescription.h"
#include "ModuleDescription.h"
#include "ObjectDescriptionManager.h"

namespace megamol::core::factories {

/**
 * Class of rendering graph module description manager
 */
class ModuleDescriptionManager : public ObjectDescriptionManager<ModuleDescription> {
public:
    /** ctor */
    ModuleDescriptionManager() : ObjectDescriptionManager<megamol::core::factories::ModuleDescription>() {}

    /** dtor */
    ~ModuleDescriptionManager() override = default;

    /* deleted copy ctor */
    ModuleDescriptionManager(const ModuleDescriptionManager& src) = delete;

    /* deleted assignment operator */
    ModuleDescriptionManager& operator=(const ModuleDescriptionManager& rhs) = delete;

    /**
     * Registers a module description
     *
     * @param Cp The ModuleDescription class
     */
    template<class Cp>
    void RegisterDescription() {
        this->Register(std::make_shared<const Cp>());
    }

    /**
     * Registers a module using a module auto description
     *
     * @param Cp The Module class
     */
    template<class Cp>
    void RegisterAutoDescription() {
        this->RegisterDescription<ModuleAutoDescription<Cp>>();
    }
};

} // namespace megamol::core::factories

#endif // MEGAMOLCORE_FACTORIES_MODULEDESCRIPTIONMANAGER_H_INCLUDED
