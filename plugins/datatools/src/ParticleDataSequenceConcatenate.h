/*
 * ParticleDataSequenceConcatenate.h
 *
 * Copyright (C) 2015 by TU Dresden (S. Grottel)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

namespace megamol::datatools {

class ParticleDataSequenceConcatenate : public core::Module {
public:
    static const char* ClassName() {
        return "ParticleDataSequenceConcatenate";
    }
    static const char* Description() {
        return "Concatenates data from two MultiParticleList data source modules";
    }
    static bool IsAvailable() {
        return true;
    }
    ParticleDataSequenceConcatenate();
    ~ParticleDataSequenceConcatenate() override;

protected:
    bool create() override;
    void release() override;

private:
    bool getExtend(megamol::core::Call& c);
    bool getData(megamol::core::Call& c);
    core::CalleeSlot dataOutSlot;
    core::CallerSlot dataIn1Slot;
    core::CallerSlot dataIn2Slot;
};

} // namespace megamol::datatools
