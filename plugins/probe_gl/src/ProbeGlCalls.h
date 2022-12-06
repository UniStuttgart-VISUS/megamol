/*
 * ProbeGlCalls.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef PROBE_GL_CALLS_H_INCLUDED
#define PROBE_GL_CALLS_H_INCLUDED

#include "mmstd/generic/CallGeneric.h"

#include "ProbeInteractionCollection.h"

namespace megamol {
namespace probe_gl {

class CallProbeInteraction
        : public core::GenericVersionedCall<std::shared_ptr<ProbeInteractionCollection>, core::Spatial3DMetaData> {
public:
    inline CallProbeInteraction()
            : GenericVersionedCall<std::shared_ptr<ProbeInteractionCollection>, core::Spatial3DMetaData>() {}
    ~CallProbeInteraction() = default;

    static const char* ClassName(void) {
        return "CallProbeInteraction";
    }
    static const char* Description(void) {
        return "Call that transports a probe interaction collection";
    }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<CallProbeInteraction> CallProbeInteractionDescription;

} // namespace probe_gl
} // namespace megamol


#endif // !PROBE_GL_CALLS_H_INCLUDED
