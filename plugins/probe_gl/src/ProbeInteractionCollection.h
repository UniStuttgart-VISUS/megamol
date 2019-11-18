/*
 * ProbeInteractionCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef PROBE_INTERACTION_COLLECTION_H_INCLUDED
#define PROBE_INTERACTION_COLLECTION_H_INCLUDED

namespace megamol {
namespace probe_gl {

enum InteractionType { MOVE, SELECT, DESELECT, HIGHLIGHT };

struct ProbeManipulation {
    InteractionType type;
    uint32_t        obj_id;
};

class ProbeInteractionCollection {
public:

    ProbeInteractionCollection();
    ~ProbeInteractionCollection();

private:
};

}
}

#endif // !PROBE_INTERACTION_COLLECTION_H_INCLUDED
