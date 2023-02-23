/**
 * MegaMol
 * Copyright (c) 2017, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/generic/CallGeneric.h"

#include "LightCollection.h"

namespace megamol::core::view::light {

class CallLight : public core::GenericVersionedCall<LightCollection, core::EmptyMetaData> {
public:
    CallLight() = default;
    ~CallLight() override = default;

    static const char* ClassName() {
        return "CallLight";
    }
    static const char* Description() {
        return "Call that transports a collection of lights";
    }
};

typedef core::factories::CallAutoDescription<CallLight> CallLightDescription;

} // namespace megamol::core::view::light
