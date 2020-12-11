#include "stdafx.h"
#include "ComputeDistance.h"

#include "DTW.hpp"

#include "glm/glm.hpp"

#include "mmstd_datatools/misc/FrechetDistance.h"
//#include "mmstd_datatools/misc/KDE.h"


megamol::probe::ComputeDistance::ComputeDistance()
        : _out_table_slot("outTable", ""), _in_probes_slot("inProbes", "") {
    _out_table_slot.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(0), &ComputeDistance::get_data_cb);
    _out_table_slot.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(1), &ComputeDistance::get_extent_cb);
    MakeSlotAvailable(&_out_table_slot);

    _in_probes_slot.SetCompatibleCall<CallProbesDescription>();
    MakeSlotAvailable(&_in_probes_slot);
}


megamol::probe::ComputeDistance::~ComputeDistance() {
    this->Release();
}


bool megamol::probe::ComputeDistance::create() {
    return true;
}


void megamol::probe::ComputeDistance::release() {}


bool megamol::probe::ComputeDistance::get_data_cb(core::Call& c) {
    auto out_table = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (out_table == nullptr)
        return false;
    auto in_probes = _in_probes_slot.CallAs<CallProbes>();
    if (in_probes == nullptr)
        return false;

    if (!(*in_probes)(CallProbes::CallGetMetaData))
        return false;
    if (!(*in_probes)(CallProbes::CallGetData))
        return false;

    auto const& meta_data = in_probes->getMetaData();

    if (in_probes->hasUpdate() || meta_data.m_frame_ID != _frame_id) {
        auto const& probe_data = in_probes->getData();
        auto const probe_count = probe_data->getProbeCount();

        _row_count = probe_count;
        _col_count = probe_count;
        _col_infos.clear();
        _col_infos.resize(probe_count);
        _dis_mat.clear();
        _dis_mat.resize(probe_count * probe_count, 0.0f);

        if (probe_count == 0)
            return false;

        bool vec_probe = false;
        {
            auto test_probe = probe_data->getGenericProbe(0);
            vec_probe = std::holds_alternative<Vec4Probe>(test_probe);
        }

        if (vec_probe) {
            core::utility::log::Log::DefaultLog.WriteInfo("[ComputeDistance] Computing distances for vector probes");
#pragma omp parallel for
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                std::vector<glm::vec4> a_samples;
                {
                    auto const a_probe = probe_data->getProbe<Vec4Probe>(a_pidx);
                    auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                    a_samples.resize(a_samples_tmp.size());
                    std::transform(a_samples_tmp.begin(), a_samples_tmp.end(), a_samples.begin(),
                        [](auto const& el) { return glm::vec4(el[0], el[1], el[2], el[3]); });
                    std::for_each(a_samples.begin(), a_samples.end(), [](auto& el) { el = glm::normalize(el); });
                }
                for (std::int64_t b_pidx = a_pidx; b_pidx < probe_count; ++b_pidx) {
                    std::vector<glm::vec4> b_samples;
                    {
                        auto const b_probe = probe_data->getProbe<Vec4Probe>(b_pidx);
                        auto const& b_samples_tmp = b_probe.getSamplingResult()->samples;
                        b_samples.resize(b_samples_tmp.size());
                        std::transform(b_samples_tmp.begin(), b_samples_tmp.end(), b_samples.begin(),
                            [](auto const& el) { return glm::vec4(el[0], el[1], el[2], el[3]); });
                        std::for_each(b_samples.begin(), b_samples.end(), [](auto& el) { el = glm::normalize(el); });
                    }
                    std::vector<float> scores(a_samples.size());
                    std::transform(a_samples.begin(), a_samples.end(), b_samples.begin(), scores.begin(),
                        [](auto const& lhs, auto const& rhs) {
                            if (std::isnan(lhs.x) || std::isnan(lhs.y) || std::isnan(lhs.z) || std::isnan(lhs.w) ||
                                std::isnan(rhs.x) || std::isnan(rhs.y) || std::isnan(rhs.z) || std::isnan(rhs.w)) {
                                return 0.0f;
                            }
                            // return glm::dot(lhs, rhs);
                            auto const angle = std::acos(glm::dot(lhs, rhs));
                            auto const angle_dis = angle / 3.14f;
                            return 1.0f - angle_dis;
                        });
                    auto const score = std::accumulate(scores.begin(), scores.end(), 0.0f, std::plus<float>());
                    _dis_mat[a_pidx + b_pidx * probe_count] = score;
                    _dis_mat[b_pidx + a_pidx * probe_count] = score;
                }
            }
        } else {
            core::utility::log::Log::DefaultLog.WriteInfo("[ComputeDistance] Computing distances for scalar probes");
            std::size_t base_skip = 0;
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                auto const a_probe = probe_data->getProbe<FloatProbe>(a_pidx);
                auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                auto first_not_nan = std::find_if_not(a_samples_tmp.begin(), a_samples_tmp.end(), std::isnan<float>);
                if (first_not_nan != a_samples_tmp.end()) {
                    auto first_not_nan_idx = std::distance(a_samples_tmp.begin(), first_not_nan);
                    if (first_not_nan_idx > base_skip) {
                        base_skip = first_not_nan_idx;
                    }
                }
            }
#pragma omp parallel for
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                /*std::vector<std::vector<double>> a_samples;
                {
                    auto const a_probe = probe_data->getProbe<FloatProbe>(a_pidx);
                    auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                    a_samples.resize(a_samples_tmp.size());
                    double tmp_cnt = 0.0;
                    std::transform(
                        a_samples_tmp.cbegin(), a_samples_tmp.cend(), a_samples.begin(), [&tmp_cnt](auto const val) {
                            return std::vector<double>{tmp_cnt++, static_cast<double>(val)};
                        });
                    auto const it = std::stable_partition(
                        a_samples.begin(), a_samples.end(), [](auto const& el) { return !std::isnan(el[1]); });
                    std::for_each(it, a_samples.end(), [](auto& el) { el[1] = 0.0; });
                }*/
                std::vector<double> a_samples;
                {
                    auto const a_probe = probe_data->getProbe<FloatProbe>(a_pidx);
                    auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                    a_samples.resize(a_samples_tmp.size());
                    std::transform(
                        a_samples_tmp.cbegin(), a_samples_tmp.cend(), a_samples.begin(), [](auto const val) {
                            return static_cast<double>(val);
                        });
                    /*auto const it = std::stable_partition(
                        a_samples.begin(), a_samples.end(), [](auto const& el) { return !std::isnan(el); });
                    std::for_each(it, a_samples.end(), [](auto& el) { el = 0.0; });*/
                    /*auto first_val = std::find_if_not(a_samples.begin(), a_samples.end(), std::isnan<double>);
                    auto last_val = std::find_if_not(a_samples.rbegin(), a_samples.rend(), std::isnan<double>);
                    std::for_each(a_samples.begin(), first_val, [&first_val](auto& el){
                        el = *first_val;});
                    std::for_each(a_samples.rbegin(), last_val, [&last_val](auto& el) { el = *last_val; });
                    auto a_tmp = a_samples;
                    for (size_t idx = 0; idx < a_samples.size(); ++idx) {
                        auto a = idx - 1;
                        if (idx == 0)
                            a = 0;
                        auto b = idx;
                        auto c = idx + 1;
                        if (idx == a_samples.size() - 1)
                            c = a_samples.size() - 1;

                        a_samples[idx] = 0.5 * (0.5 * a_tmp[a] + a_tmp[b] + 0.5 * a_tmp[c]);
                    }*/
                    a_samples.erase(a_samples.begin(), a_samples.begin() + base_skip);
                }
                for (std::int64_t b_pidx = a_pidx; b_pidx < probe_count; ++b_pidx) {
                    /*std::vector<std::vector<double>> b_samples;
                    {
                        auto const b_probe = probe_data->getProbe<FloatProbe>(b_pidx);
                        auto const b_samples_tmp = b_probe.getSamplingResult()->samples;
                        b_samples.resize(b_samples_tmp.size());
                        double tmp_cnt = 0.0;
                        std::transform(b_samples_tmp.cbegin(), b_samples_tmp.cend(), b_samples.begin(),
                            [&tmp_cnt](auto const val) {
                                return std::vector<double>{tmp_cnt++, static_cast<double>(val)};
                            });
                        auto const it = std::stable_partition(
                            b_samples.begin(), b_samples.end(), [](auto const& el) { return !std::isnan(el[1]); });
                        std::for_each(it, b_samples.end(), [](auto& el) { el[1] = 0.0; });
                    }*/
                    std::vector<double> b_samples;
                    {
                        auto const b_probe = probe_data->getProbe<FloatProbe>(b_pidx);
                        auto const b_samples_tmp = b_probe.getSamplingResult()->samples;
                        b_samples.resize(b_samples_tmp.size());
                        std::transform(b_samples_tmp.cbegin(), b_samples_tmp.cend(), b_samples.begin(),
                            [](auto const val) {
                                return static_cast<double>(val);
                            });
                        /*auto const it = std::stable_partition(
                            b_samples.begin(), b_samples.end(), [](auto const& el) { return !std::isnan(el); });
                        std::for_each(it, b_samples.end(), [](auto& el) { el = 0.0; });*/
                        /*auto first_val = std::find_if_not(b_samples.begin(), b_samples.end(), std::isnan<double>);
                        auto last_val = std::find_if_not(b_samples.rbegin(), b_samples.rend(), std::isnan<double>);
                        std::for_each(b_samples.begin(), first_val, [&first_val](auto& el) { el = *first_val; });
                        std::for_each(b_samples.rbegin(), last_val, [&last_val](auto& el) { el = *last_val; });
                        auto b_tmp = b_samples;
                        for (size_t idx = 0; idx < b_samples.size(); ++idx) {
                            auto a = idx - 1;
                            if (idx == 0)
                                a = 0;
                            auto b = idx;
                            auto c = idx + 1;
                            if (idx == b_samples.size() - 1)
                                c = b_samples.size() - 1;

                            b_samples[idx] = 0.5 * (0.5 * b_tmp[a] + b_tmp[b] + 0.5 * b_tmp[c]);
                        }*/
                        b_samples.erase(b_samples.begin(), b_samples.begin() + base_skip);
                    }
                    auto const dis = stdplugin::datatools::misc::frechet_distance<double>(a_samples, b_samples);
                    //auto const dis = DTW::dtw_distance_only(a_samples, b_samples, 2);
                    _dis_mat[a_pidx + b_pidx * probe_count] = dis;
                    _dis_mat[b_pidx + a_pidx * probe_count] = dis;
                }
            }
        }

        for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
            auto const minmax =
                std::minmax_element(&_dis_mat[a_pidx * probe_count], &_dis_mat[a_pidx * probe_count + probe_count]);
            _col_infos[a_pidx].SetName("p" + std::to_string(a_pidx));
            _col_infos[a_pidx].SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
            _col_infos[a_pidx].SetMinimumValue(*minmax.first);
            _col_infos[a_pidx].SetMaximumValue(*minmax.second);
        }

        _frame_id = meta_data.m_frame_ID;
        ++_out_data_hash;
    }

    out_table->SetFrameCount(meta_data.m_frame_cnt);
    out_table->SetFrameID(_frame_id);
    out_table->SetDataHash(_out_data_hash);
    out_table->Set(_col_count, _row_count, _col_infos.data(), _dis_mat.data());

    return true;
}


bool megamol::probe::ComputeDistance::get_extent_cb(core::Call& c) {
    auto ctd = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (ctd == nullptr)
        return false;

    auto cpd = _in_probes_slot.CallAs<CallProbes>();
    if (cpd == nullptr)
        return false;

    // get metadata from probes
    auto meta_data = cpd->getMetaData();
    meta_data.m_frame_ID = ctd->GetFrameID();
    cpd->setMetaData(meta_data);

    if (!(*cpd)(1))
        return false;

    // put metadata in table call
    meta_data = cpd->getMetaData();
    ctd->SetFrameCount(meta_data.m_frame_cnt);

    return true;
}
