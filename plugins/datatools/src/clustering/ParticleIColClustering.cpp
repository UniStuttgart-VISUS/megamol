#include "datatools/clustering/ParticleIColClustering.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"


megamol::datatools::clustering::ParticleIColClustering::ParticleIColClustering()
        : AbstractParticleManipulator("outData", "inData")
        , _eps_slot("eps", "")
        , _minpts_slot("minpts", "")
        , _icol_weight("icol weight", "") {
    _eps_slot << new core::param::FloatParam(0.1f, 0.0f, 1.0f);
    MakeSlotAvailable(&_eps_slot);

    _minpts_slot << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&_minpts_slot);

    _icol_weight << new core::param::FloatParam(0.5f, 0.0f, 1.0f);
    MakeSlotAvailable(&_icol_weight);
}


megamol::datatools::clustering::ParticleIColClustering::~ParticleIColClustering() {
    this->Release();
}


bool megamol::datatools::clustering::ParticleIColClustering::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    if (_frame_id != inData.FrameID() || _in_data_hash != inData.DataHash() || isDirty()) {
        outData = inData;

        auto const pl_count = outData.GetParticleListCount();

        auto const p_bbox = outData.AccessBoundingBoxes().ObjectSpaceBBox();

        auto const eps = _eps_slot.Param<core::param::FloatParam>()->Value();
        auto const minpts = static_cast<index_t>(_minpts_slot.Param<core::param::IntParam>()->Value());
        auto const icol_weight = _icol_weight.Param<core::param::FloatParam>()->Value();

        std::array<float, 4> weights = {(1.0f - icol_weight), (1.0f - icol_weight), (1.0f - icol_weight), icol_weight};

        _points.resize(pl_count);
        _kd_trees.resize(pl_count);
        _ret_cols.resize(pl_count);

        for (std::remove_const_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
            auto& parts = outData.AccessParticles(pl_idx);

            if (parts.GetVertexDataType() == geocalls::SimpleSphericalParticles::VERTDATA_NONE ||
                (parts.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I &&
                    parts.GetColourDataType() != geocalls::SimpleSphericalParticles::COLDATA_DOUBLE_I)) {
                continue;
            }

            auto const p_count = parts.GetCount();

            if (_frame_id != inData.FrameID() || _in_data_hash != inData.DataHash()) {
                // rebuild search structure

                std::vector<float> cur_points(p_count * 4);

                auto const xAcc = parts.GetParticleStore().GetXAcc();
                auto const yAcc = parts.GetParticleStore().GetYAcc();
                auto const zAcc = parts.GetParticleStore().GetZAcc();
                auto const iAcc = parts.GetParticleStore().GetCRAcc();

                for (std::remove_const_t<decltype(p_count)> pidx = 0; pidx < p_count; ++pidx) {
                    cur_points[pidx * 4 + 0] = xAcc->Get_f(pidx);
                    cur_points[pidx * 4 + 1] = yAcc->Get_f(pidx);
                    cur_points[pidx * 4 + 2] = zAcc->Get_f(pidx);
                    cur_points[pidx * 4 + 3] = iAcc->Get_f(pidx);
                }

                std::array<float, 8> bbox = {p_bbox.GetLeft(), p_bbox.GetRight(), p_bbox.GetBottom(), p_bbox.GetTop(),
                    p_bbox.GetBack(), p_bbox.GetFront(), parts.GetMinColourIndexValue(),
                    parts.GetMaxColourIndexValue()};

                _points[pl_idx] = std::make_shared<genericPointcloud<float, 4>>(cur_points, bbox, weights);
                _points[pl_idx]->normalize_data();

                _kd_trees[pl_idx] = std::make_shared<kd_tree_t<float, 4>>(
                    4, *_points[pl_idx], nanoflann::KDTreeSingleIndexAdaptorParams());
                _kd_trees[pl_idx]->buildIndex();
            }

            auto const cluster_res = DBSCAN(_kd_trees[pl_idx], eps * eps, minpts);

            _ret_cols[pl_idx].resize(p_count);
            std::transform(cluster_res.cbegin(), cluster_res.cend(), _ret_cols[pl_idx].begin(),
                [](auto const val) { return static_cast<float>(val); });

            parts.SetColourData(geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I, _ret_cols[pl_idx].data());
            auto const minmax = std::minmax_element(_ret_cols[pl_idx].cbegin(), _ret_cols[pl_idx].cend());
            parts.SetColourMapIndexValues(*minmax.first, *minmax.second);

            auto const num_clusters = static_cast<index_t>(*minmax.second) - 1;
            core::utility::log::Log::DefaultLog.WriteInfo(
                "[ParticleIColClustering]: Number of clusters in list idx %d = %d", pl_idx, num_clusters);
            core::utility::log::Log::DefaultLog.WriteInfo(
                "[ParticleIColClustering]: Min idx %f; Max idx %f", *minmax.first, *minmax.second);
        }

        _frame_id = inData.FrameID();
        _in_data_hash = inData.DataHash();
        resetDirty();
        ++_out_data_hash;
    }

    outData.SetDataHash(_out_data_hash);
    outData.SetUnlocker(inData.GetUnlocker());
    inData.SetUnlocker(nullptr, false);

    return true;
}
