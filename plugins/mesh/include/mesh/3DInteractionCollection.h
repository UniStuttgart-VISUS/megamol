/*
 * 3DInteractionCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef THREE_DIMENSIONAL_INTERACTION_COLLECTION_H_INCLUDED
#define THREE_DIMENSIONAL_INTERACTION_COLLECTION_H_INCLUDED

#include <map>
#include <queue>
#include <vector>

namespace megamol {
namespace mesh {

enum InteractionType { MOVE_ALONG_AXIS, MOVE_IN_PLANE, ROTATE_AROUND_AXIS, SELECT, DESELET, HIGHLIGHT };

struct ThreeDimensionalInteraction {
    InteractionType type;
    uint32_t        obj_id;
    float           axis_x;
    float           axis_y;
    float           axis_z;
    float           origin_x;
    float           origin_y;
    float           origin_z;
};

struct ThreeDimensionalManipulation {
    InteractionType type;
    uint32_t        obj_id;
    float           axis_x;
    float           axis_y;
    float           axis_Z;
    float           value;
};


class ThreeDimensionalInteractionCollection {
public:
    ThreeDimensionalInteractionCollection() = default;
    ~ThreeDimensionalInteractionCollection() = default;

    void addInteractionObject(uint32_t obj_id, std::vector<ThreeDimensionalInteraction> const& interactions)
    {
        m_available_interactions.insert({obj_id, interactions});
    }

    std::vector<ThreeDimensionalInteraction> getAvailableInteractions(uint32_t obj_id){

        std::vector<ThreeDimensionalInteraction> retval;

        auto query = m_available_interactions.find(obj_id);

        if (query != m_available_interactions.end()){
            retval = query->second;
        }

        return retval;
    }

    std::queue<ThreeDimensionalManipulation>& accessPendingManipulations(){
        return m_pending_manipulations;
    }

private:

    std::map<uint32_t, std::vector<ThreeDimensionalInteraction>> m_available_interactions;
    std::queue<ThreeDimensionalManipulation>                     m_pending_manipulations;
   
};

}
}

#endif // !3D_INTERACTION_COLLECTION_H_INCLUDED
