/*
 * MPDCListsConcatenate.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

namespace megamol::datatools {

class MPDCListsConcatenate : public core::Module {
public:
    static const char* ClassName() {
        return "MPDCListsConcatenate";
    }
    static const char* Description() {
        return "Concatenates the particle lists from two MPDCs";
    }
    static bool IsAvailable() {
        return true;
    }
    MPDCListsConcatenate();
    ~MPDCListsConcatenate() override;

protected:
    bool create() override;
    void release() override;

private:
    bool getExtent(megamol::core::Call& c);
    bool getData(megamol::core::Call& c);
    core::CalleeSlot dataOutSlot;
    core::CallerSlot dataIn1Slot;
    core::CallerSlot dataIn2Slot;
}; // end class MPDCListsConcatenate

} // namespace megamol::datatools
