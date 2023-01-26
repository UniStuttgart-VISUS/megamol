#pragma once

#include "mmcore/param/ParamSlot.h"

#include "datatools/AbstractParticleManipulator.h"

#include "datatools/clustering/DBSCAN.h"

namespace megamol::datatools::clustering {
class ParticleIColClustering : public AbstractParticleManipulator {
public:
    static const char* ClassName(void) {
        return "ParticleIColClustering";
    }

    static const char* Description(void) {
        return "Clusters particles according to position and ICol";
    }

    static bool IsAvailable(void) {
        return true;
    }

    ParticleIColClustering(void);

    ~ParticleIColClustering(void) override;

protected:
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    bool isDirty() {
        return _eps_slot.IsDirty() || _minpts_slot.IsDirty() || _icol_weight.IsDirty();
    }

    void resetDirty() {
        _eps_slot.ResetDirty();
        _minpts_slot.ResetDirty();
        _icol_weight.ResetDirty();
    }

    core::param::ParamSlot _eps_slot;

    core::param::ParamSlot _minpts_slot;

    core::param::ParamSlot _icol_weight;

    std::vector<std::shared_ptr<genericPointcloud<float, 4>>> _points;

    std::vector<std::shared_ptr<kd_tree_t<float, 4>>> _kd_trees;

    std::vector<std::vector<float>> _ret_cols;

    unsigned int _frame_id = std::numeric_limits<unsigned int>::max();

    std::size_t _in_data_hash = std::numeric_limits<std::size_t>::max();

    std::size_t _out_data_hash = 0;
};
} // namespace megamol::datatools::clustering
