/*
 * EventCall.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_EVENT_CALL_H_INCLUDED
#define MEGAMOL_EVENT_CALL_H_INCLUDED

#include <memory>

#include "DoubleBufferedEventCollection.h"
#include "mmstd/generic/CallGeneric.h"

namespace megamol {
namespace core {

class CallEvent
        : public core::GenericVersionedCall<std::shared_ptr<DoubleBufferedEventCollection>, core::EmptyMetaData> {
public:
    CallEvent() = default;
    ~CallEvent() = default;

    static const char* ClassName(void) {
        return "CallEvent";
    }
    static const char* Description(void) {
        return "Call that transports a DoubleBufferedEventCollection for read-write access";
    }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<CallEvent> CallEventDescription;

} // namespace core
} // namespace megamol

#endif // !MEGAMOL_EVENT_CALL_H_INCLUDED
