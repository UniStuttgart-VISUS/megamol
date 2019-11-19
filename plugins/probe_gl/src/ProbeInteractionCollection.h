/*
 * ProbeInteractionCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef PROBE_INTERACTION_COLLECTION_H_INCLUDED
#define PROBE_INTERACTION_COLLECTION_H_INCLUDED

#include <queue>

namespace megamol {
namespace probe_gl {

enum InteractionType { SELECT, DESELECT, HIGHLIGHT, MOVE, PLACE, REMOVE };

struct ProbeManipulation {
    InteractionType type;
    uint32_t        obj_id;
    float           x;
    float           y;
    float           z;
};

class ProbeInteractionCollection {
public:

    ProbeInteractionCollection() = default;
    ~ProbeInteractionCollection() = default;

    std::queue<ProbeManipulation>& accessPendingManipulations() { return m_pending_manipulations; }

private:
    std::queue<ProbeManipulation> m_pending_manipulations;
};

}
}

#endif // !PROBE_INTERACTION_COLLECTION_H_INCLUDED
