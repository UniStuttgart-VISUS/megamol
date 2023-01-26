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

namespace megamol {
namespace datatools {

class MPDCListsConcatenate : public core::Module {
public:
    static const char* ClassName(void) {
        return "MPDCListsConcatenate";
    }
    static const char* Description(void) {
        return "Concatenates the particle lists from two MPDCs";
    }
    static bool IsAvailable(void) {
        return true;
    }
    MPDCListsConcatenate();
    ~MPDCListsConcatenate() override;

protected:
    bool create(void) override;
    void release(void) override;

private:
    bool getExtent(megamol::core::Call& c);
    bool getData(megamol::core::Call& c);
    core::CalleeSlot dataOutSlot;
    core::CallerSlot dataIn1Slot;
    core::CallerSlot dataIn2Slot;
}; // end class MPDCListsConcatenate

} // end namespace datatools
} // end namespace megamol
