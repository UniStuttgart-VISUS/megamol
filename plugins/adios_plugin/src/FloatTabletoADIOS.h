/*
 * FloatTabletoADIOS.h
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
    
class FloatTabletoADIOS : public core::Module {
    
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "FloatTabletoADIOS"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Converts MegaMols CallFloatTableData into ADIOS-based IO."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    FloatTabletoADIOS(void);

    /** Dtor. */
    virtual ~FloatTabletoADIOS(void);

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
    std::string cleanUpColumnHeader(const vislib::TString &header) const;

private:

    core::CallerSlot ftSlot;
    core::CalleeSlot adiosSlot;

    adiosDataMap dataMap;

    std::map<std::string, size_t> columnIndex;
};

} // end namespace adios
} // end namespace megamol