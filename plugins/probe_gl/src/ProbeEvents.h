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

struct ClearSelection : public core::EventCollection::AbstractEvent {
    ClearSelection(size_t frame_id) : AbstractEvent(frame_id) {}
};

struct ProbeHighlight : public core::EventCollection::AbstractEvent {
    ProbeHighlight(size_t frame_id, uint32_t obj_id) : AbstractEvent(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct ProbeDehighlight : public core::EventCollection::AbstractEvent {
    ProbeDehighlight(size_t frame_id, uint32_t obj_id) : AbstractEvent(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct Select : public core::EventCollection::AbstractEvent {
    Select(size_t frame_id, uint32_t obj_id) : AbstractEvent(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct Deselect : public core::EventCollection::AbstractEvent {
    Deselect(size_t frame_id, uint32_t obj_id) : AbstractEvent(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct ToggleShowGlyphs : public core::EventCollection::AbstractEvent {
    ToggleShowGlyphs(size_t frame_id) : AbstractEvent(frame_id) {}
};

struct ToggleShowHull : public core::EventCollection::AbstractEvent {
    ToggleShowHull(size_t frame_id) : AbstractEvent(frame_id) {}
};

struct ToggleShowProbes : public core::EventCollection::AbstractEvent {
    ToggleShowProbes(size_t frame_id) : AbstractEvent(frame_id) {}
};

} // namespace probe_gl
} // namespace megamol

#endif // !PROBE_EVENTS_H_INCLUDED
