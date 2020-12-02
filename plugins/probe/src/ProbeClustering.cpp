#include "stdafx.h"
#include "ProbeClustering.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "mmstd_datatools/table/TableDataCall.h"

#include "ProbeCalls.h"


megamol::probe::ProbeClustering::ProbeClustering()
        : _out_probes_slot("outProbes", "")
        , _in_probes_slot("inProbes", "")
        , _in_table_slot("inTable", "")
        , _eps_slot("eps", "")
        , _minpts_slot("minpts", "")
        , _threshold_slot("threshold", "")
        , _handwaving_slot("handwaving", "") {
    _out_probes_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(0), &ProbeClustering::get_data_cb);
    _out_probes_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(1), &ProbeClustering::get_extent_cb);
    MakeSlotAvailable(&_out_probes_slot);

    _in_probes_slot.SetCompatibleCall<CallProbesDescription>();
    MakeSlotAvailable(&_in_probes_slot);

    _in_table_slot.SetCompatibleCall<stdplugin::datatools::table::TableDataCallDescription>();
    MakeSlotAvailable(&_in_table_slot);

    _eps_slot << new core::param::FloatParam(0.1f, 0.0f);
    MakeSlotAvailable(&_eps_slot);

    _minpts_slot << new core::param::IntParam(1, 0);
    MakeSlotAvailable(&_minpts_slot);

    _threshold_slot << new core::param::FloatParam(0.1f, 0.0f);
    MakeSlotAvailable(&_threshold_slot);

    _handwaving_slot << new core::param::FloatParam(0.05f, 0.0f);
    MakeSlotAvailable(&_handwaving_slot);
}


megamol::probe::ProbeClustering::~ProbeClustering() {
    this->Release();
}


bool megamol::probe::ProbeClustering::create() {
    return true;
}


void megamol::probe::ProbeClustering::release() {}


bool megamol::probe::ProbeClustering::get_data_cb(core::Call& c) {
    auto out_probes = dynamic_cast<CallProbes*>(&c);
    if (out_probes == nullptr)
        return false;
    auto in_probes = _in_probes_slot.CallAs<CallProbes>();
    if (in_probes == nullptr)
        return false;
    auto in_table = _in_table_slot.CallAs<stdplugin::datatools::table::TableDataCall>();
    if (in_table == nullptr)
        return false;

    if (!(*in_probes)(CallProbes::CallGetMetaData))
        return false;
    if (!(*in_probes)(CallProbes::CallGetData))
        return false;
    if (!(*in_table)(1))
        return false;
    if (!(*in_table)(0))
        return false;

    auto const& meta_data = in_probes->getMetaData();
    auto& probes = in_probes->getData();

    if (in_probes->hasUpdate() || meta_data.m_frame_ID != _frame_id || in_table->DataHash() != _in_table_data_hash ||
        is_dirty()) {
        auto const num_probes = probes->getProbeCount();

        auto const eps = _eps_slot.Param<core::param::FloatParam>()->Value();
        auto const minpts = _minpts_slot.Param<core::param::IntParam>()->Value();
        auto const threshold = _threshold_slot.Param<core::param::FloatParam>()->Value();
        auto const handwaving = _handwaving_slot.Param<core::param::FloatParam>()->Value();

        std::vector<float> cur_points(num_probes * 3);
        for (std::remove_const_t<decltype(num_probes)> pidx = 0; pidx < num_probes; ++pidx) {
            auto const probe = probes->getProbe<FloatProbe>(pidx);
            cur_points[pidx * 3 + 0] = probe.m_position[0];
            cur_points[pidx * 3 + 1] = probe.m_position[1];
            cur_points[pidx * 3 + 2] = probe.m_position[2];
        }

        auto const p_bbox = meta_data.m_bboxs.BoundingBox();
        std::array<float, 6> bbox = {p_bbox.GetLeft(), p_bbox.GetRight(), p_bbox.GetBottom(), p_bbox.GetTop(),
            p_bbox.GetBack(), p_bbox.GetFront()};
        _points = std::make_shared<stdplugin::datatools::genericPointcloud<float, 3>>(
            cur_points, bbox, std::array<float, 3>{1.0f, 1.0f, 1.0f});
        _points->normalize_data();
        _kd_tree = std::make_shared<stdplugin::datatools::clustering::kd_tree_t<float, 3>>(
            3, *_points, nanoflann::KDTreeSingleIndexAdaptorParams());
        _kd_tree->buildIndex();

        auto const col_count = in_table->GetColumnsCount();
        auto const row_count = in_table->GetRowsCount();
        auto const sim_matrix = in_table->GetData();

        /*auto const cluster_res =
            stdplugin::datatools::clustering::DBSCAN_with_similarity<float, 3>(_kd_tree, eps, minpts,
                [sim_matrix, col_count, row_count, threshold](
                    stdplugin::datatools::clustering::index_t a, stdplugin::datatools::clustering::index_t b) -> bool {
                    auto const val = sim_matrix[a + b * col_count];
                    return val <= threshold;
                });*/
        auto const cluster_res = stdplugin::datatools::clustering::DBSCAN_with_similarity_and_score<float, 3>(
            _kd_tree, eps, minpts,
            [sim_matrix, col_count, row_count, threshold](
                stdplugin::datatools::clustering::index_t a, stdplugin::datatools::clustering::index_t b) -> bool {
                auto const val = sim_matrix[a + b * col_count];
                return val <= threshold;
            },
            [sim_matrix, col_count, row_count, handwaving](
                stdplugin::datatools::clustering::index_t pivot,
                std::vector<stdplugin::datatools::clustering::index_t> const& cluster)
                -> stdplugin::datatools::clustering::index_t {
                if (cluster.empty())
                    return pivot;
                std::vector<float> scores;
                scores.reserve(cluster.size());
                for (auto const& lhs : cluster) {
                    auto val = 0.0f;
                    for (auto const& rhs : cluster) {
                        val = sim_matrix[lhs + rhs * col_count];
                    }
                    val /= static_cast<float>(cluster.size() - 1);
                    scores.push_back(val);
                }
                auto it = std::min_element(scores.begin(), scores.end());
                auto idx = std::distance(scores.begin(), it);
                return cluster[idx];
            });


        /*[sim_matrix, col_count, row_count](stdplugin::datatools::clustering::index_t a,
            stdplugin::datatools::clustering::search_res_t<float> const& vec) -> float {
            std::vector<float> tmp_vals(vec.size());
            std::transform(vec.cbegin(), vec.cend(), tmp_vals.begin(),
                [&sim_matrix, col_count, a](auto const& el) { return sim_matrix[el.first + a * col_count]; });
            auto const sum = std::accumulate(tmp_vals.cbegin(), tmp_vals.cend(), 0.0f, std::plus<float>());
            return sum / static_cast<float>(vec.size());
        }*/

        for (decltype(cluster_res)::size_type pidx = 0; pidx < cluster_res.size(); ++pidx) {
            auto probe = probes->getProbe<FloatProbe>(pidx);
            probe.m_cluster_id = cluster_res[pidx];
            probes->setProbe(pidx, probe);
        }

        auto const max_el = std::max_element(cluster_res.cbegin(), cluster_res.cend());

        core::utility::log::Log::DefaultLog.WriteInfo("[ProbeClustering] Num Clusters %d", (*max_el) - 1);

        _frame_id = meta_data.m_frame_ID;
        _in_table_data_hash = in_table->DataHash();
        ++_out_data_hash;
        reset_dirty();
    }

    out_probes->setData(probes, _out_data_hash);
    out_probes->setMetaData(meta_data);

    return true;
}


bool megamol::probe::ProbeClustering::get_extent_cb(core::Call& c) {
    auto cp = dynamic_cast<CallProbes*>(&c);
    if (cp == nullptr)
        return false;

    auto ct = this->_in_table_slot.CallAs<stdplugin::datatools::table::TableDataCall>();
    if (ct == nullptr)
        return false;
    auto cprobes = this->_in_probes_slot.CallAs<CallProbes>();
    if (cprobes == nullptr)
        return false;


    if (!(*cprobes)(1))
        return false;
    auto meta_data = cprobes->getMetaData();
    if (!(*ct)(1))
        return false;


    // put metadata in mesh call
    cp->setMetaData(meta_data);

    return true;
}
