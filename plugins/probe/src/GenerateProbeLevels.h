/*
 * GenerateProbeLevels.h
 * Copyright (C) 2022 by MegaMol Team
 * Alle Rechte vorbehalten.
 */


#pragma once

#include "glm/glm.hpp"

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mesh/MeshCalls.h"
#include "mmcore/param/ParamSlot.h"
#include "probe/CallKDTree.h"
#include "probe/ProbeCollection.h"
#include "PlaceProbes.h"

#include "probe/PBCkdTree.h"
#include "DualMeshProbeSampling.h"


namespace megamol {
namespace probe {

inline glm::vec2 to_vec2(std::array<float, 2> input) {
    return glm::vec2(input[0], input[1]);
}

class GenerateProbeLevels : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "GenerateProbeLevels";
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

    GenerateProbeLevels();
    virtual ~GenerateProbeLevels();

protected:
    virtual bool create();
    virtual void release();

    uint32_t _version;

    core::CallerSlot _probe_rhs_slot;
    core::CalleeSlot _probe_lhs_slot;

private:
    float getAvgDist();
    bool getData(core::Call& call);
    bool getMetaData(core::Call& call);
    bool parameterChanged(core::param::ParamSlot& p);
    bool calcSphericalCoordinates();
    bool calcMercatorProjection();
    bool calcLevels(std::shared_ptr<ProbeCol> inputProbes);
    std::array<float, 3> calcInverseMercatorProjection(std::array<float, 2> const& coords, float const& r);
    std::array<float, 3> calcInverseSphericalProjection(std::array<float, 3> const& coords);
    bool doRelaxation();

    std::vector<ProbeCol::ProbeLevel> _levels;
    bool _recalc = false;

    std::shared_ptr<my_kd_tree_t> _probe_tree;
    std::shared_ptr<const data2KD> _probe_dataKD;

    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PBC_adaptor>, PBC_adaptor,
        2 /* dim */>
        my_2d_kd_tree_t;

    std::shared_ptr<my_2d_kd_tree_t> _probe_mercator_tree;
    std::shared_ptr<const PBC_adaptor> _probe_mercator_dataKD;

    std::vector<std::array<float, 3>> _probe_positions;
    std::vector<std::array<float, 3>> _probe_positions_spherical_coordinates;
    std::vector<std::array<float, 2>> _probe_positions_mercator;
    std::array<float,4> _mercator_bounds;
    std::array<float,3> _center;
};

} // namespace probe
} // namespace megamol
