/*
* SyncDataSourcesCall.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* Alle Rechte vorbehalten.
*/

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/Call.h"

namespace megamol {
namespace core {
namespace cluster {

class MEGAMOLCORE_API SyncDataSourcesCall : public core::Call {
public:
   /**
    * Answer the name of the objects of this description.
    *
    * @return The name of the objects of this description.
    */
    static const char *ClassName(void) {
        return "SyncDataSourcesCall";
    }

    /**
    * Gets a human readable description of the module.
    *
    * @return A human readable description of the module.
    */
    static const char *Description(void) {
        return "Call for sync data sources";
    }

    /**
    * Answer the number of functions used for this call.
    *
    * @return The number of functions used for this call.
    */
    static unsigned int FunctionCount(void) {
        return 2;
    }

    /**
    * Answer the name of the function used for this call.
    *
    * @param idx The index of the function to return it's name.
    *
    * @return The name of the requested function.
    */
    static const char * FunctionName(unsigned int idx) {
        switch (idx) {
        case 0: return "checkDirty";
        case 1: return "setDirty";
        default: return NULL;
        }
    }

    /** Ctor. */
    SyncDataSourcesCall();

    /** Dtor. */
    virtual ~SyncDataSourcesCall(void);

    bool getFilenameDirty() { return fnameDirty; }
    void resetFilenameDirty() { this->fnameDirty = false; }
    void setFilenameDirty() { this->fnameDirty = true; }


private:

    bool fnameDirty;

}; // end class SyncDataSourcesCall
typedef core::factories::CallAutoDescription<SyncDataSourcesCall> SyncDataSourcesCallDescription;
} // namespace cluster
} // namespace core
} // namespace megamol