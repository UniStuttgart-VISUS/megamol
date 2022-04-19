/*
 * CallLight.h
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef CALL_LIGHTS_H_INCLUDED
#define CALL_LIGHTS_H_INCLUDED

#include "mmcore/CallGeneric.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "LightCollection.h"

namespace megamol {
namespace core {
namespace view {
namespace light {

class CallLight : public core::GenericVersionedCall<LightCollection, core::EmptyMetaData> {
public:
    CallLight() = default;
    ~CallLight() = default;

    static const char* ClassName(void) {
        return "CallLight";
    }
    static const char* Description(void) {
        return "Call that transports a collection of lights";
    }
};

typedef core::factories::CallAutoDescription<CallLight> CallLightDescription;

} // namespace light
} // namespace view
} // namespace core
} // namespace megamol

#endif //!CALL_LIGHTS_H_INCLUDED
