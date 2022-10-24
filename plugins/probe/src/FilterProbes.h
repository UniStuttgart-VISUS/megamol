/*
 * FilterProbes.h
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */


#pragma once

#include "glm/glm.hpp"

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mesh/MeshCalls.h"
#include "mmcore/param/ParamSlot.h"
#include "probe/ProbeCollection.h"
#include "probe/CallKDTree.h"


namespace megamol {
namespace probe {

inline glm::vec3 to_vec3(std::array<float,3> const& input) {
    return glm::vec3(input[0], input[1], input[2]);
}


class FilterProbes : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "FilterProbes";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "...";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    FilterProbes();
    virtual ~FilterProbes();

protected:
    virtual bool create();
    virtual void release();

    uint32_t _version;

    core::param::ParamSlot _center_param;
    core::CallerSlot _probe_rhs_slot;
    core::CalleeSlot _probe_lhs_slot;

private:
    typedef kd_adaptor<std::vector<std::array<float, 3>>> data2KD;
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, data2KD>, data2KD, 3> my_kd_tree_t;

    bool getData(core::Call& call);
    bool getMetaData(core::Call& call);
    bool parameterChanged(core::param::ParamSlot& p);
    float calculateAverageDistance(std::vector<std::array<float,3>> const& input_data, int const num_neighbors);
    float getDistance(std::array<float, 3> const& point1, std::array<float, 3> const& point2);
    bool newPosDir(int const id_filtered, int const id_all);

    std::shared_ptr<ProbeCol> _filtered_probe_collection;

    bool _recalc;
};

} // namespace probe
} // namespace megamol
