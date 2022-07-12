/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>

#include "DoubleBufferedEventCollection.h"
#include "mmstd/generic/CallGeneric.h"

namespace megamol::core {

class CallEvent
        : public core::GenericVersionedCall<std::shared_ptr<DoubleBufferedEventCollection>, core::EmptyMetaData> {
public:
    CallEvent() = default;
    ~CallEvent() = default;

    static const char* ClassName() {
        return "CallEvent";
    }
    static const char* Description() {
        return "Call that transports a DoubleBufferedEventCollection for read-write access";
    }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<CallEvent> CallEventDescription;

} // namespace megamol::core
