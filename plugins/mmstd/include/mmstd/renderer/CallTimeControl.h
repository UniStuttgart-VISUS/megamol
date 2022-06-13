/*
 * CallTimeControl.h
 *
 * Copyright (C) 2012 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLTIMECONTROL_H_INCLUDED
#define MEGAMOLCORE_CALLTIMECONTROL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/renderer/TimeControl.h"


namespace megamol {
namespace core {
namespace view {


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
    static const char* ClassName(void) {
        return "CallTimeControl";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call connecting time control objects";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
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
    CallTimeControl(void);

    /** Dtor. */
    virtual ~CallTimeControl(void);

    /**
     * Answer the master time control
     *
     * @return The master time control
     */
    inline TimeControl* Master(void) const {
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


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLTIMECONTROL_H_INCLUDED */
