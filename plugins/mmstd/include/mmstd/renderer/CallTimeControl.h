/**
 * MegaMol
 * Copyright (c) 2012, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/renderer/TimeControl.h"

namespace megamol::core::view {

/**
 * Call connecting time control objects
 */
class CallTimeControl : public Call {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "CallTimeControl";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call connecting time control objects";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
        return 1;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "getMaster";
        default:
            return NULL;
        }
    }

    /** Ctor. */
    CallTimeControl();

    /** Dtor. */
    virtual ~CallTimeControl();

    /**
     * Answer the master time control
     *
     * @return The master time control
     */
    inline TimeControl* Master() const {
        return this->m;
    }

    /**
     * sets the master time control
     *
     * @param m The master time control
     */
    inline void SetMaster(TimeControl* m) {
        this->m = m;
    }

private:
    /** The master time control */
    TimeControl* m;
};


/** Description class typedef */
typedef factories::CallAutoDescription<CallTimeControl> CallTimeControlDescription;

} // namespace megamol::core::view
