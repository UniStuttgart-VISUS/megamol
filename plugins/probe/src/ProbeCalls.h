/*
 * ProbeCalls.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef PROBE_CALLS_H_INCLUDED
#define PROBE_CALLS_H_INCLUDED

#include "mmcore/CallGeneric.h"

#include "ProbeCollection.h"

namespace megamol {
namespace probe {

class PROBE_API CallProbes
    : public core::CallGeneric<std::shared_ptr<ProbeCollection>, core::BasicMetaData> {
public:
    inline CallProbes() : CallGeneric<std::shared_ptr<ProbeCollection>, core::BasicMetaData>() {}
    ~CallProbes() = default;

    static const char* ClassName(void) { return "CallProbes"; }
    static const char* Description(void) { return "Call that transports a probe collection"; }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<CallProbes> CallProbesDescription;

}
}


#endif // !PROBE_CALLS_H_INCLUDED
