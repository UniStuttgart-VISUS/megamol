/*
 * TickSwitch.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol {
namespace core {
namespace job {

/**
 * Module for propagating a tick.
 *
 * @author Alexander Straub
 */
class TickSwitch : public core::Module {
public:
    /**
     * Human-readable class name
     */
    static const char* ClassName() {
        return "TickSwitch";
    }

    /**
     * Human-readable class description
     */
    static const char* Description() {
        return "Module for propagating a tick";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    /**
     * Ctor.
     */
    TickSwitch();

    /**
     * Dtor.
     */
    ~TickSwitch() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override {
        return true;
    }

    /**
     * Implementation of 'Release'.
     */
    void release() override {}

private:
    /** Callback for ticks */
    bool TickCallback(core::Call& call);

    /** Slot from where this module gets its ticks (=output) */
    core::CalleeSlot incoming_slot;

    /** Slot to where this module puts its ticks (=input) */
    core::CallerSlot outgoing_slot_1;
    core::CallerSlot outgoing_slot_2;
    core::CallerSlot outgoing_slot_3;
    core::CallerSlot outgoing_slot_4;
};

} // namespace job
} // namespace core
} // namespace megamol
