/*
 * ProbeEvents.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmstd/event/EventCollection.h"

namespace megamol {
namespace probe_gl {

struct ProbeHighlight : public core::EventCollection::Event<false> {
    ProbeHighlight(size_t frame_id, uint32_t obj_id) : Event(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct ProbeDehighlight : public core::EventCollection::Event<false> {
    ProbeDehighlight(size_t frame_id, uint32_t obj_id) : Event<false>(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct ProbeSelect : public core::EventCollection::Event<false> {
    ProbeSelect(size_t frame_id, uint32_t obj_id) : Event<false>(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct ProbeDeselect : public core::EventCollection::Event<false> {
    ProbeDeselect(size_t frame_id, uint32_t obj_id) : Event<false>(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct ProbeSelectExclusive : public core::EventCollection::Event<false> {
    ProbeSelectExclusive(size_t frame_id, uint32_t obj_id) : Event<false>(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct ProbeSelectToggle : public core::EventCollection::Event<false> {
    ProbeSelectToggle(size_t frame_id, uint32_t obj_id) : Event<false>(frame_id), obj_id(obj_id) {}
    uint32_t obj_id;
};

struct ProbeClearSelection : public core::EventCollection::Event<false> {
    ProbeClearSelection(size_t frame_id) : Event<false>(frame_id) {}
};

struct DataFilterByProbeSelection : public core::EventCollection::Event<false> {
    DataFilterByProbeSelection(size_t frame_id) : Event<false>(frame_id) {}
};

struct DataFilterByProbingDepth : public core::EventCollection::Event<false> {
    DataFilterByProbingDepth(size_t frame_id, float depth) : Event<false>(frame_id), depth(depth) {}
    float depth;
};

struct DataClearFilter : public core::EventCollection::Event<false> {
    DataClearFilter(size_t frame_id) : Event<false>(frame_id) {}
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
