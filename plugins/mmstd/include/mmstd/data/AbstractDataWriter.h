/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/DataWriterCtrlCall.h"


namespace megamol::core {

/**
 * Abstract base class for data writer modules
 */
class AbstractDataWriter : public Module {
public:
    /** Ctor. */
    AbstractDataWriter();

    /** Dtor. */
    ~AbstractDataWriter() override;

protected:
    /**
     * The main function
     *
     * @return True on success
     */
    virtual bool run() = 0;

    /**
     * Function querying the writers capabilities
     *
     * @param call The call to receive the capabilities
     *
     * @return True on success
     */
    virtual bool getCapabilities(DataWriterCtrlCall& call) = 0;

    /**
     * Called to abort the run function.
     *
     * @return True on success
     */
    virtual bool abort();

private:
    /**
     * Event handler for incoming run calls
     *
     * @param call The incoming call
     *
     * @return True on success
     */
    bool onCallRun(Call& call);

    /**
     * Event handler for incoming run calls
     *
     * @param call The incoming call
     *
     * @return True on success
     */
    bool onCallGetCapability(Call& call);

    /**
     * Event handler for incoming run calls
     *
     * @param call The incoming call
     *
     * @return True on success
     */
    bool onCallAbort(Call& call);

    /**
     * Manual start of the run method.
     *
     * @param slot Must be the triggerButtonSlot
     */
    bool triggerManualRun(param::ParamSlot& slot);

    /** The slot for incoming control commands */
    CalleeSlot controlSlot;

    /** Triggers execution of the 'run' method */
    param::ParamSlot manualRunSlot;
};

} // namespace megamol::core
