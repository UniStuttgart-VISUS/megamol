/*
 * DataSetTimeRewriteModule.h
 *
 * Copyright (C) 2014 by CGV TU Dresden
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "PluginsResource.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/factories/CallDescription.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Cuboid.h"


namespace megamol {
namespace datatools {


/**
 * In-Between management module to change time codes of a data set
 */
class DataSetTimeRewriteModule : public core::Module {
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
    static const char* ClassName(void) {
        return "DataSetTimeRewriteModule";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "In-Between management module to change time codes of a data set";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    DataSetTimeRewriteModule(void);

    /** Dtor. */
    virtual ~DataSetTimeRewriteModule(void);

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
    /**
     * Tests if th description of a call seems compatible
     *
     * @param desc The description to test
     *
     * @return True if description seems compatible
     */
    static bool IsCallDescriptionCompatible(core::factories::CallDescription::ptr desc);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& caller);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getExtentCallback(core::Call& caller);

    /**
     * Checks if the callee and the caller slot are connected with the
     * same call classes
     *
     * @param outCall The incoming call requesting data
     *
     * @return True if everything is fine.
     */
    bool checkConnections(core::Call* outCall);

    /** The slot for publishing data to the writer */
    core::CalleeSlot outDataSlot;

    /** The slot for requesting data from the source */
    core::CallerSlot inDataSlot;

    /** The number of the first frame */
    core::param::ParamSlot firstFrameSlot;

    /** The number of the last frame */
    core::param::ParamSlot lastFrameSlot;

    /** The step length between two frames */
    core::param::ParamSlot frameStepSlot;
};

} /* end namespace datatools */
} /* end namespace megamol */
