/*
 * ProbeInteractionCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef PROBE_INTERACTION_COLLECTION_H_INCLUDED
#define PROBE_INTERACTION_COLLECTION_H_INCLUDED

#include <list>

namespace megamol {
namespace probe_gl {

enum InteractionType { SELECT, DESELECT, HIGHLIGHT, DEHIGHLIGHT,  MOVE, PLACE, REMOVE };

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

    std::list<ProbeManipulation>& accessPendingManipulations() { return m_pending_manipulations; }

private:
    std::list<ProbeManipulation> m_pending_manipulations;
};

}
}

#endif // !PROBE_INTERACTION_COLLECTION_H_INCLUDED
