/*
 * SyncDataSourcesCall.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol::core::cluster {

class SyncDataSourcesCall : public core::Call {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "SyncDataSourcesCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call for sync data sources";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
        return 2;
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
            return "checkDirty";
        case 1:
            return "setDirty";
        default:
            return NULL;
        }
    }

    /** Ctor. */
    SyncDataSourcesCall();

    /** Dtor. */
    ~SyncDataSourcesCall() override;

    bool getFilenameDirty() {
        return fnameDirty;
    }
    void resetFilenameDirty() {
        this->fnameDirty = false;
    }
    void setFilenameDirty() {
        this->fnameDirty = true;
    }


private:
    bool fnameDirty;

}; // end class SyncDataSourcesCall
typedef core::factories::CallAutoDescription<SyncDataSourcesCall> SyncDataSourcesCallDescription;
} // namespace megamol::core::cluster
