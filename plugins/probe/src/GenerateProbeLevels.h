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


namespace megamol {
namespace probe {


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
    bool calcLevels(std::shared_ptr<ProbeCollection> inputProbes);
    std::array<float, 3> calcInverseMercatorProjection(std::array<float, 2> const& coords, float const& r);
    std::array<float, 3> calcInverseSphericalProjection(std::array<float, 3> const& coords);

    std::vector<ProbeCollection::ProbeLevel> _levels;
    bool _recalc = false;

    std::shared_ptr<my_kd_tree_t> _probe_tree;
    std::shared_ptr<const data2KD> _probe_dataKD;
    std::vector<std::array<float, 3>> _probe_positions;
    std::vector<std::array<float, 3>> _probe_positions_spherical_coodrinates;
    std::vector<std::array<float, 2>> _probe_positions_mercator;
    std::vector<float> _mercator_bounds;
    std::array<float,3> _center;
};

} // namespace probe
} // namespace megamol
