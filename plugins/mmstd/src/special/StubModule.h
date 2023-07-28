/**
 * MegaMol
 * Copyright (c) 2017, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "PluginsResource.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

namespace megamol::core::special {

/**
 * Simple module accepting all inbound and outbound call classes.
 * This module can be used as a stub for debugging and test purposes for development modules
 * for which suitable sinks do not exist.
 */
class StubModule : public Module {
public:
    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        Module::requested_lifetime_resources(req);
        req.require<frontend_resources::PluginsResource>();
    }

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "StubModule";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Stub module which accepts all ingoing (inSlot) and outgoing (outSlot) calls "
               "for debugging and test purposes.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** ctor */
    StubModule();

    /** dtor */
    ~StubModule() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

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

} // namespace megamol::core::special
