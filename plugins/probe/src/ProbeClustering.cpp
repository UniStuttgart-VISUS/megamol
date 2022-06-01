#include "ProbeClustering.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "datatools/table/TableDataCall.h"


megamol::probe::ProbeClustering::ProbeClustering()
        : _out_probes_slot("outProbes", "")
        , _in_probes_slot("inProbes", "")
        , _in_table_slot("inTable", "")
        , _eps_slot("eps", "")
        , _minpts_slot("minpts", "")
        , _threshold_slot("threshold", "")
        , _handwaving_slot("handwaving", "")
        , _lhs_idx_slot("debug::lhs_idx", "")
        , _rhs_idx_slot("debug::rhs_idx", "")
        , _print_debug_info_slot("debug::print", "")
        , _toggle_reps_slot("toggle reps", "")
        , _angle_threshold_slot("angle threshold", "") {
    _out_probes_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(0), &ProbeClustering::get_data_cb);
    _out_probes_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(1), &ProbeClustering::get_extent_cb);
    MakeSlotAvailable(&_out_probes_slot);

    _in_probes_slot.SetCompatibleCall<CallProbesDescription>();
    MakeSlotAvailable(&_in_probes_slot);

    _in_table_slot.SetCompatibleCall<datatools::table::TableDataCallDescription>();
    MakeSlotAvailable(&_in_table_slot);

    _eps_slot << new core::param::FloatParam(0.1f, 0.0f);
    MakeSlotAvailable(&_eps_slot);

    _minpts_slot << new core::param::IntParam(1, 0);
    MakeSlotAvailable(&_minpts_slot);

    _threshold_slot << new core::param::FloatParam(0.1f, 0.0f);
    MakeSlotAvailable(&_threshold_slot);

    _handwaving_slot << new core::param::FloatParam(0.05f, 0.0f);
    MakeSlotAvailable(&_handwaving_slot);

    _lhs_idx_slot << new core::param::IntParam(0, 0);
    MakeSlotAvailable(&_lhs_idx_slot);

    _rhs_idx_slot << new core::param::IntParam(0, 0);
    MakeSlotAvailable(&_rhs_idx_slot);

    _print_debug_info_slot << new core::param::ButtonParam();
    _print_debug_info_slot.SetUpdateCallback(&ProbeClustering::print_debug_info);
    MakeSlotAvailable(&_print_debug_info_slot);

    _toggle_reps_slot << new core::param::BoolParam(true);
    MakeSlotAvailable(&_toggle_reps_slot);

    _angle_threshold_slot << new core::param::FloatParam(45.0f, 0.0f);
    MakeSlotAvailable(&_angle_threshold_slot);
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
    auto in_table = _in_table_slot.CallAs<datatools::table::TableDataCall>();
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

    _col_count = in_table->GetColumnsCount();
    _row_count = in_table->GetRowsCount();
    _sim_matrix = in_table->GetData();

    auto const& meta_data = in_probes->getMetaData();
    _probes = in_probes->getData();

    // if (is_debug_dirty()) {
    //    auto const lhs_idx = _lhs_idx_slot.Param<core::param::IntParam>()->Value();
    //    auto const rhs_idx = _rhs_idx_slot.Param<core::param::IntParam>()->Value();
    //    if (lhs_idx < _col_count && rhs_idx < _row_count) {
    //        auto const val = _sim_matrix[lhs_idx + rhs_idx * _col_count];
    //        core::utility::log::Log::DefaultLog.WriteInfo(
    //            "[ProbeClustering]: Similiarty val for %d:%d is %f", lhs_idx, rhs_idx, val);
    //    }
    //    // reset_debug_dirty();

    //    if (lhs_idx < _probes->getProbeCount() && rhs_idx < _probes->getProbeCount()) {

    //        auto rhs_generic_probe = _probes->getGenericProbe(rhs_idx);
    //        auto lhs_generic_probe = _probes->getGenericProbe(lhs_idx);
    //        uint32_t rhs_cluster_id = std::visit(
    //            [](auto&& arg) -> uint32_t { return static_cast<uint32_t>(arg.m_cluster_id); }, rhs_generic_probe);
    //        uint32_t lhs_cluster_id = std::visit(
    //            [](auto&& arg) -> uint32_t { return static_cast<uint32_t>(arg.m_cluster_id); }, lhs_generic_probe);
    //        core::utility::log::Log::DefaultLog.WriteInfo(
    //            "[ProbeClustering]: Assigned cluster IDs for %d:%d are %d:%d", lhs_idx, rhs_idx, lhs_cluster_id,
    //            rhs_cluster_id);
    //    }
    //}

    if (in_probes->hasUpdate() || meta_data.m_frame_ID != _frame_id || in_table->DataHash() != _in_table_data_hash ||
        is_dirty() || _toggle_reps_slot.IsDirty() /*|| is_debug_dirty()*/) {
        if (in_probes->hasUpdate() || meta_data.m_frame_ID != _frame_id ||
            in_table->DataHash() != _in_table_data_hash || is_dirty() /*|| is_debug_dirty()*/) {
            auto const num_probes = _probes->getProbeCount();

            auto const eps = _eps_slot.Param<core::param::FloatParam>()->Value();
            auto const minpts = _minpts_slot.Param<core::param::IntParam>()->Value();
            auto const threshold = _threshold_slot.Param<core::param::FloatParam>()->Value();
            auto const handwaving = _handwaving_slot.Param<core::param::FloatParam>()->Value();
            auto const angle_threshold = glm::radians(_angle_threshold_slot.Param<core::param::FloatParam>()->Value());

            bool vec_probe = false;
            bool distrib_probe = false;
            {
                auto const test_probe = _probes->getGenericProbe(0);
                vec_probe = std::holds_alternative<Vec4Probe>(test_probe);
                distrib_probe = std::holds_alternative<FloatDistributionProbe>(test_probe);
            }

            std::vector<float> cur_points(num_probes * 3);
            _cur_dirs.resize(num_probes);
            if (vec_probe) {
                for (std::remove_const_t<decltype(num_probes)> pidx = 0; pidx < num_probes; ++pidx) {
                    auto const probe = _probes->getProbe<Vec4Probe>(pidx);
                    cur_points[pidx * 3 + 0] = probe.m_position[0];
                    cur_points[pidx * 3 + 1] = probe.m_position[1];
                    cur_points[pidx * 3 + 2] = probe.m_position[2];
                    _cur_dirs[pidx] = glm::vec3(probe.m_direction[0], probe.m_direction[1], probe.m_direction[2]);
                }
            } else if (distrib_probe) {
                for (std::remove_const_t<decltype(num_probes)> pidx = 0; pidx < num_probes; ++pidx) {
                    auto const probe = _probes->getProbe<FloatDistributionProbe>(pidx);
                    cur_points[pidx * 3 + 0] = probe.m_position[0];
                    cur_points[pidx * 3 + 1] = probe.m_position[1];
                    cur_points[pidx * 3 + 2] = probe.m_position[2];
                    _cur_dirs[pidx] = glm::vec3(probe.m_direction[0], probe.m_direction[1], probe.m_direction[2]);
                }
            } else {
                for (std::remove_const_t<decltype(num_probes)> pidx = 0; pidx < num_probes; ++pidx) {
                    auto const probe = _probes->getProbe<FloatProbe>(pidx);
                    cur_points[pidx * 3 + 0] = probe.m_position[0];
                    cur_points[pidx * 3 + 1] = probe.m_position[1];
                    cur_points[pidx * 3 + 2] = probe.m_position[2];
                    _cur_dirs[pidx] = glm::vec3(probe.m_direction[0], probe.m_direction[1], probe.m_direction[2]);
                }
            }

            /*if (is_debug_dirty()) {
                auto const lhs_idx = _lhs_idx_slot.Param<core::param::IntParam>()->Value();
                auto const rhs_idx = _rhs_idx_slot.Param<core::param::IntParam>()->Value();
                if (lhs_idx < _col_count && rhs_idx < _row_count) {
                    auto const angle =
                        glm::degrees(std::acos(glm::dot(glm::normalize(_cur_dirs[lhs_idx]),
            glm::normalize(_cur_dirs[rhs_idx])))); core::utility::log::Log::DefaultLog.WriteInfo(
                        "[ProbeClustering]: Angle between %d:%d is %f", lhs_idx, rhs_idx, angle);
                }
            }*/

            if (in_probes->hasUpdate() || meta_data.m_frame_ID != _frame_id ||
                in_table->DataHash() != _in_table_data_hash || is_dirty()) {
                auto const p_bbox = meta_data.m_bboxs.BoundingBox();
                std::array<float, 6> bbox = {p_bbox.GetLeft(), p_bbox.GetRight(), p_bbox.GetBottom(), p_bbox.GetTop(),
                    p_bbox.GetBack(), p_bbox.GetFront()};
                _points = std::make_shared<datatools::genericPointcloud<float, 3>>(
                    cur_points, bbox, std::array<float, 3>{1.0f, 1.0f, 1.0f});
                //_points->normalize_data();
                _kd_tree = std::make_shared<datatools::clustering::kd_tree_t<float, 3>>(
                    3, *_points, nanoflann::KDTreeSingleIndexAdaptorParams());
                _kd_tree->buildIndex();


                /*auto const cluster_res =
                    datatools::clustering::DBSCAN_with_similarity<float, 3>(_kd_tree, eps, minpts,
                        [sim_matrix, col_count, row_count, threshold](
                            datatools::clustering::index_t a, datatools::clustering::index_t b) ->
                   bool { auto const val = sim_matrix[a + b * col_count]; return val <= threshold;
                        });*/
                _cluster_res = datatools::clustering::GROWING_with_similarity_and_score<float, 3>(
                    _kd_tree, eps * eps, minpts,
                    [this, threshold, angle_threshold](
                        datatools::clustering::index_t a, datatools::clustering::index_t b) -> bool {
                        auto const val = _sim_matrix[a + b * _col_count];
                        auto const crit_a = val <= threshold;

                        auto const a_dir = _cur_dirs[a];
                        auto const b_dir = _cur_dirs[b];
                        auto const rad_angle = std::acos(glm::dot(glm::normalize(a_dir), glm::normalize(b_dir)));
                        auto const crit_b = rad_angle <= angle_threshold;

                        return crit_a && crit_b;
                    },
                    [this, handwaving](datatools::clustering::index_t pivot,
                        std::vector<datatools::clustering::index_t> const& cluster) -> datatools::clustering::index_t {
                        if (cluster.empty())
                            return pivot;
                        std::vector<float> scores;
                        scores.reserve(cluster.size());
                        for (auto const& lhs : cluster) {
                            auto val = 0.0f;
                            for (auto const& rhs : cluster) {
                                val += _sim_matrix[lhs + rhs * _col_count];
                            }
                            val /= static_cast<float>(cluster.size() - 1);
                            scores.push_back(val);
                        }
                        auto it = std::min_element(scores.begin(), scores.end());
                        auto idx = std::distance(scores.begin(), it);
                        return cluster[idx];
                    });


                /*[sim_matrix, col_count, row_count](datatools::clustering::index_t a,
                    datatools::clustering::search_res_t<float> const& vec) -> float {
                    std::vector<float> tmp_vals(vec.size());
                    std::transform(vec.cbegin(), vec.cend(), tmp_vals.begin(),
                        [&sim_matrix, col_count, a](auto const& el) { return sim_matrix[el.first + a * col_count]; });
                    auto const sum = std::accumulate(tmp_vals.cbegin(), tmp_vals.cend(), 0.0f, std::plus<float>());
                    return sum / static_cast<float>(vec.size());
                }*/

                if (vec_probe) {
                    for (decltype(_cluster_res)::size_type pidx = 0; pidx < _cluster_res.size(); ++pidx) {
                        auto probe = _probes->getProbe<Vec4Probe>(pidx);
                        probe.m_cluster_id = _cluster_res[pidx];
                        _probes->setProbe(pidx, probe);
                    }
                } else if (distrib_probe) {
                    for (decltype(_cluster_res)::size_type pidx = 0; pidx < _cluster_res.size(); ++pidx) {
                        auto probe = _probes->getProbe<FloatDistributionProbe>(pidx);
                        probe.m_cluster_id = _cluster_res[pidx];
                        _probes->setProbe(pidx, probe);
                    }
                } else {
                    for (decltype(_cluster_res)::size_type pidx = 0; pidx < _cluster_res.size(); ++pidx) {
                        auto probe = _probes->getProbe<FloatProbe>(pidx);
                        probe.m_cluster_id = _cluster_res[pidx];
                        _probes->setProbe(pidx, probe);
                    }
                }
            }
        }
        auto const max_el = std::max_element(_cluster_res.cbegin(), _cluster_res.cend());

        core::utility::log::Log::DefaultLog.WriteInfo("[ProbeClustering] Num Clusters %d", (*max_el) - 1);


        bool toggle_reps = _toggle_reps_slot.Param<core::param::BoolParam>()->Value();

        if (toggle_reps) {
            auto const num_probes = _probes->getProbeCount();
            // reset representant flag
            bool vec_probe = false;
            bool distrib_probe = false;
            bool float_probe = false;
            {
                auto const test_probe = _probes->getGenericProbe(0);
                vec_probe = std::holds_alternative<Vec4Probe>(test_probe);
                distrib_probe = std::holds_alternative<FloatDistributionProbe>(test_probe);
                float_probe = std::holds_alternative<FloatProbe>(test_probe);
            }
            if (vec_probe) {
                for (int i = 0; i < num_probes; ++i) {
                    auto probe = _probes->getProbe<Vec4Probe>(i);
                    probe.m_representant = false;
                    _probes->setProbe(i, probe);
                }
            } else if (distrib_probe) {
                for (int i = 0; i < num_probes; ++i) {
                    auto probe = _probes->getProbe<FloatDistributionProbe>(i);
                    probe.m_representant = false;
                    _probes->setProbe(i, probe);
                }
            } else if (float_probe) {
                for (int i = 0; i < num_probes; ++i) {
                    auto probe = _probes->getProbe<FloatProbe>(i);
                    probe.m_representant = false;
                    _probes->setProbe(i, probe);
                }
            } else {
                for (int i = 0; i < num_probes; ++i) {
                    auto probe = _probes->getProbe<IntProbe>(i);
                    probe.m_representant = false;
                    _probes->setProbe(i, probe);
                }
            }


            std::unordered_map<datatools::clustering::index_t, std::vector<datatools::clustering::index_t>> cluster_map;
            cluster_map.reserve(*max_el);

            for (decltype(_cluster_res)::size_type pidx = 0; pidx < _cluster_res.size(); ++pidx) {
                cluster_map[_cluster_res[pidx]].push_back(pidx);
            }

            std::vector<datatools::clustering::index_t> cluster_reps;
            cluster_reps.reserve(*max_el);
            for (auto const& el : cluster_map) {
                datatools::clustering::index_t min_idx = 0;
                float min_score = std::numeric_limits<float>::max();
                for (auto const& idx : el.second) {
                    auto const current_idx = idx;
                    for (auto const& tmp_idx : el.second) {
                        if (tmp_idx == current_idx)
                            continue;
                        auto const val = _sim_matrix[current_idx + tmp_idx * _col_count];
                        if (val < min_score) {
                            min_idx = current_idx;
                            min_score = val;
                        }
                    }
                }
                cluster_reps.push_back(min_idx);
            }

            if (vec_probe) {
                for (auto const& el : cluster_reps) {
                    auto probe = _probes->getProbe<Vec4Probe>(el);
                    probe.m_representant = true;
                    _probes->setProbe(el, probe);
                }
            } else if (distrib_probe) {
                for (auto const& el : cluster_reps) {
                    auto probe = _probes->getProbe<FloatDistributionProbe>(el);
                    probe.m_representant = true;
                    _probes->setProbe(el, probe);
                }
            } else if (float_probe) {
                for (auto const& el : cluster_reps) {
                    auto probe = _probes->getProbe<FloatProbe>(el);
                    probe.m_representant = true;
                    _probes->setProbe(el, probe);
                }
            } else {
                for (auto const& el : cluster_reps) {
                    auto probe = _probes->getProbe<IntProbe>(el);
                    probe.m_representant = true;
                    _probes->setProbe(el, probe);
                }
            }
            /*std::vector<char> indicator(probes->getProbeCount(), 1);
            for (auto const& el : cluster_reps) {
                indicator[el] = 0;
            }
            probes->erase_probes(indicator);*/
        }

        _frame_id = meta_data.m_frame_ID;
        _in_table_data_hash = in_table->DataHash();
        ++_out_data_hash;
        reset_dirty();
        _toggle_reps_slot.ResetDirty();
    }
    /*if (is_debug_dirty()) {
        reset_debug_dirty();
    }*/

    out_probes->setData(_probes, _out_data_hash);
    out_probes->setMetaData(meta_data);

    return true;
}


bool megamol::probe::ProbeClustering::get_extent_cb(core::Call& c) {
    auto cp = dynamic_cast<CallProbes*>(&c);
    if (cp == nullptr)
        return false;

    auto ct = this->_in_table_slot.CallAs<datatools::table::TableDataCall>();
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
