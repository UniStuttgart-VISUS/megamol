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


namespace megamol::datatools {

class DumpIColorHistogramModule : public megamol::core::Module {
public:
    static const char* ClassName() {
        return "DumpIColorHistogramModule";
    }
    static const char* Description() {
        return "Dump (DEBUG! DO NOT USE)";
    }
    static bool IsAvailable() {
        return true;
    }
    DumpIColorHistogramModule();
    ~DumpIColorHistogramModule() override;
    bool create() override;
    void release() override;

private:
    bool dump(::megamol::core::param::ParamSlot& param);
    megamol::core::CallerSlot inDataSlot;
    megamol::core::param::ParamSlot dumpBtnSlot;
    megamol::core::param::ParamSlot timeSlot;
};


} // namespace megamol::datatools
