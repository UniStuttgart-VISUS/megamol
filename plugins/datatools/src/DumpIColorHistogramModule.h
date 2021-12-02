/*
 * DumpIColorHistogramModule.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace datatools {

class DumpIColorHistogramModule : public megamol::core::Module {
public:
    static const char* ClassName(void) {
        return "DumpIColorHistogramModule";
    }
    static const char* Description(void) {
        return "Dump (DEBUG! DO NOT USE)";
    }
    static bool IsAvailable(void) {
        return true;
    }
    DumpIColorHistogramModule(void);
    virtual ~DumpIColorHistogramModule(void);
    virtual bool create(void);
    virtual void release(void);

private:
    bool dump(::megamol::core::param::ParamSlot& param);
    megamol::core::CallerSlot inDataSlot;
    megamol::core::param::ParamSlot dumpBtnSlot;
    megamol::core::param::ParamSlot timeSlot;
};


} /* end namespace datatools */
} /* end namespace megamol */
