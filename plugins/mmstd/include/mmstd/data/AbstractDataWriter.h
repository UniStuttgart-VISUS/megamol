/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

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

private:
    /**
     * Manual start of the run method.
     *
     * @param slot Must be the triggerButtonSlot
     */
    bool triggerManualRun(param::ParamSlot& slot);

    /** Triggers execution of the 'run' method */
    param::ParamSlot manualRunSlot;
};

} // namespace megamol::core
