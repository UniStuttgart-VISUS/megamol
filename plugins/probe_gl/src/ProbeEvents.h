/*
 * ProbeEvents.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef PROBE_EVENTS_H_INCLUDED
#define PROBE_EVENTS_H_INCLUDED

#include "mmcore/EventCollection.h"

namespace megamol {
namespace probe_gl {

struct ProbeHighlight : public core::EventCollection::AbstractEvent {
    ProbeHighlight(size_t frame_id, uint32_t obj_id) : AbstractEvent(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct ProbeDehighlight : public core::EventCollection::AbstractEvent {
    ProbeDehighlight(size_t frame_id, uint32_t obj_id) : AbstractEvent(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

} // namespace probe_gl
} // namespace megamol

#endif // !PROBE_EVENTS_H_INCLUDED
