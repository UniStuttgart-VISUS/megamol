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

struct ClearSelection : public core::EventCollection::Event<false> {
    ClearSelection(size_t frame_id) : Event<false>(frame_id) {}
};

struct ProbeHighlight : public core::EventCollection::Event<false> {
    ProbeHighlight(size_t frame_id, uint32_t obj_id) : Event(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct ProbeDehighlight : public core::EventCollection::Event<false> {
    ProbeDehighlight(size_t frame_id, uint32_t obj_id) : Event<false>(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct Select : public core::EventCollection::Event<false> {
    Select(size_t frame_id, uint32_t obj_id) : Event<false>(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct Deselect : public core::EventCollection::Event<false> {
    Deselect(size_t frame_id, uint32_t obj_id) : Event<false>(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct ToggleShowGlyphs : public core::EventCollection::Event<false> {
    ToggleShowGlyphs(size_t frame_id) : Event<false>(frame_id) {}
};

struct ToggleShowHull : public core::EventCollection::Event<false> {
    ToggleShowHull(size_t frame_id) : Event<false>(frame_id) {}
};

struct ToggleShowProbes : public core::EventCollection::Event<false> {
    ToggleShowProbes(size_t frame_id) : Event<false>(frame_id) {}
};

} // namespace probe_gl
} // namespace megamol

#endif // !PROBE_EVENTS_H_INCLUDED
