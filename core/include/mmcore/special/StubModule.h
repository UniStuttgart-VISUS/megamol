/*
 * StubModule.h
 * Copyright (C) 2017 by MegaMol Team
 * All rights reserved. Alle Rechte vorbehalten.
 */

#ifndef MMCORE_SPECIAL_STUBMODULE_H_INCLUDED
#define MMCORE_SPECIAL_STUBMODULE_H_INCLUDED

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"

namespace megamol {
namespace core {
namespace special {

/**
 * Simple module accepting all inbound and outbound call classes.
 * This module can be used as a stub for debugging and test purposes for development modules
 * for which suitable sinks do not exist.
 */
class StubModule : public Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "StubModule";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Stub module which accepts all ingoing (inSlot) and outgoing (outSlot) calls "
            "for debugging and test purposes.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** ctor */
    StubModule(void);

    /** dtor */
    virtual ~StubModule(void);
protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);
private:
    /** Outbound connection */
    CalleeSlot outSlot;

    /** Inbound connection */
    CallerSlot inSlot;

    /**
     * Stub that calls all callback functions of the inbound connection.
     *
     * @param c Calling outbound connection
     *
     * @return True, if successful.
     */
    bool stub(Call& c);

}; /* end class StubModule */

} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif // end ifndef MMCORE_SPECIAL_STUBMODULE_H_INCLUDED
