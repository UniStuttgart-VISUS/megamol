/*
 * MultiParticletoADIOS.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "CallADIOSData.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace adios {
    
class MultiParticletoADIOS : public core::Module {
    
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "MultiParticletoADIOS"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Converts MegaMols MultiParticleDataCall into ADIOS-based IO."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    MultiParticletoADIOS(void);

    /** Dtor. */
    virtual ~MultiParticletoADIOS(void);

    bool create(void);

protected:
    void release(void);

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
    bool getHeaderCallback(core::Call& caller);

private:

    core::CallerSlot mpSlot;
    core::CalleeSlot adiosSlot;

    core::param::ParamSlot orderSlot;

    adiosDataMap dataMap;
};

} // end namespace adios
} // end namespace megamol