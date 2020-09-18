/*
 * EventCall.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_EVENT_CALL_H_INCLUDED
#define MEGAMOL_EVENT_CALL_H_INCLUDED

#include <memory>

#include "EventCollection.h"
#include "mmcore/CallGeneric.h"

namespace megamol {
namespace core {

class MEGAMOLCORE_API EventCallRead : public core::GenericVersionedCall<std::shared_ptr<EventCollection const>, core::EmptyMetaData> {
public:
    EventCallRead() = default;
    ~EventCallRead() = default;

    static const char* ClassName(void) { return "EventCallRead"; }
    static const char* Description(void) {
        return "Call that transports an EventCollection for read-only access";
    }
};

class MEGAMOLCORE_API EventCallWrite : public core::GenericVersionedCall<std::shared_ptr<EventCollection>, core::EmptyMetaData> {
public:
    EventCallWrite() = default;
    ~EventCallWrite() = default;

    static const char* ClassName(void) { return "EventCallWrite"; }
    static const char* Description(void) { return "Call that transports an EventCollection with write access"; }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<EventCallRead> EventCallReadDescription;
typedef megamol::core::factories::CallAutoDescription<EventCallWrite> EventCallWriteDescription;

}
}

#endif // !MEGAMOL_EVENT_CALL_H_INCLUDED
