/**
 * MegaMol
 * Copyright (c) 2008-2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOLCORE_FACTORIES_CALLDESCRIPTIONMANAGER_H_INCLUDED
#define MEGAMOLCORE_FACTORIES_CALLDESCRIPTIONMANAGER_H_INCLUDED
#pragma once

#include "CallAutoDescription.h"
#include "CallDescription.h"
#include "ObjectDescriptionManager.h"

namespace megamol::core {
class Call;
}

namespace megamol::core::factories {

/**
 * Class of rendering graph call description manager
 */
class CallDescriptionManager : public ObjectDescriptionManager<CallDescription> {
public:
    /** ctor */
    CallDescriptionManager() : ObjectDescriptionManager<megamol::core::factories::CallDescription>() {}

    /** dtor */
    ~CallDescriptionManager() override = default;

    /* deleted copy ctor */
    CallDescriptionManager(const CallDescriptionManager& src) = delete;

    /* deleted assignment operator */
    CallDescriptionManager& operator=(const CallDescriptionManager& rhs) = delete;

    /**
     * Registers a call description
     *
     * @param Cp The CallDescription class
     */
    template<class Cp>
    void RegisterDescription() {
        this->Register(std::make_shared<const Cp>());
    }

    /**
     * Registers a call using a module call description
     *
     * @param Cp The Call class
     */
    template<class Cp>
    void RegisterAutoDescription() {
        this->RegisterDescription<CallAutoDescription<Cp>>();
    }

    /**
     * Assignment crowbar
     *
     * @param tar The targeted object
     * @param src The source object
     *
     * @return True on success, false on failure.
     */
    bool AssignmentCrowbar(Call* tar, Call* src) const;
};

} // namespace megamol::core::factories

#endif // MEGAMOLCORE_FACTORIES_CALLDESCRIPTIONMANAGER_H_INCLUDED
