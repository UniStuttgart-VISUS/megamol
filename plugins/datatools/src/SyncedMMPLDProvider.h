/*
 * StaticMMPLDProvider.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "StaticMMPLDProvider.h"
#include "cluster/SyncDataSourcesCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace datatools {

class SyncedMMPLDProvider : public StaticMMPLDProvider {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "SyncedMMPLDProvider";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Reads a set of static MMPLDs and sychronizes the read process over multiple data sources";
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
    SyncedMMPLDProvider(void);

    /** Dtor. */
    virtual ~SyncedMMPLDProvider(void) {}

protected:
    bool create() override {
        return true;
    }

    void release() override {}

    core::CalleeSlot getSyncSlot;

private:
    bool setDirtyCallback(core::Call& c);
    bool checkDirtyCallback(core::Call& c);


}; // end class SyncedMMPLDProvider

} // namespace datatools
} // namespace megamol
