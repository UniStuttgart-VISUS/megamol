#include "ComputeDistance.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ButtonParam.h"

#include "glm/glm.hpp"

#include "datatools/misc/FrechetDistance.h"

#include <Eigen/Eigen>

#define _USE_MATH_DEFINES
#include <math.h>

//#include "frechet_distance.h"


megamol::probe_gl::ComputeDistance::ComputeDistance()
        : _out_table_slot("outTable", "")
        , _in_probes_slot("inProbes", "")
        , _stretching_factor_slot("stretching factor", "")
        , _min_sample_slot("minSample", "")
        , _max_sample_slot("maxSample", "")
        , _recalc_slot("recalc", "") {
    _out_table_slot.SetCallback(datatools::table::TableDataCall::ClassName(),
        datatools::table::TableDataCall::FunctionName(0), &ComputeDistance::get_data_cb);
    _out_table_slot.SetCallback(datatools::table::TableDataCall::ClassName(),
        datatools::table::TableDataCall::FunctionName(1), &ComputeDistance::get_extent_cb);
    MakeSlotAvailable(&_out_table_slot);

    _in_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    MakeSlotAvailable(&_in_probes_slot);

    _stretching_factor_slot << new core::param::FloatParam(5.0f, 0.0f);
    MakeSlotAvailable(&_stretching_factor_slot);

    _min_sample_slot << new core::param::IntParam(-1);
    MakeSlotAvailable(&_min_sample_slot);

    _max_sample_slot << new core::param::IntParam(-1);
    MakeSlotAvailable(&_max_sample_slot);

    _recalc_slot << new core::param::ButtonParam();
    _recalc_slot.SetUpdateCallback(&ComputeDistance::paramChanged);
    MakeSlotAvailable(&_recalc_slot);

}


megamol::probe_gl::ComputeDistance::~ComputeDistance() {
    this->Release();
}


bool megamol::probe_gl::ComputeDistance::create() {
    return true;
}


void megamol::probe_gl::ComputeDistance::release() {}


bool megamol::probe_gl::ComputeDistance::get_data_cb(core::Call& c) {
    auto out_table = dynamic_cast<datatools::table::TableDataCall*>(&c);
    if (out_table == nullptr)
        return false;
    auto in_probes = _in_probes_slot.CallAs<probe::CallProbes>();
    if (in_probes == nullptr)
        return false;

    if (!(*in_probes)(probe::CallProbes::CallGetMetaData))
        return false;
    if (!(*in_probes)(probe::CallProbes::CallGetData))
        return false;

    auto const& meta_data = in_probes->getMetaData();

    if (in_probes->hasUpdate() || meta_data.m_frame_ID != _frame_id || _stretching_factor_slot.IsDirty() || _trigger_recalc) {

        auto const& probe_data = in_probes->getData();
        auto const probe_count = probe_data->getProbeCount();

        _trigger_recalc = false;

        _row_count = probe_count;
        _col_count = probe_count;
        _col_infos.clear();
        _col_infos.resize(probe_count);
        _dis_mat.clear();
        _dis_mat.resize(probe_count * probe_count, 0.0f);

        if (probe_count == 0)
            return false;
        std::size_t sample_count = 0;
        bool vec_probe = false;
        bool distrib_probe = false;
        {
            auto test_probe = probe_data->getGenericProbe(0);
            vec_probe = std::holds_alternative<probe::Vec4Probe>(test_probe);
            distrib_probe = std::holds_alternative<probe::FloatDistributionProbe>(test_probe);
        }

        if (vec_probe) {
            core::utility::log::Log::DefaultLog.WriteInfo("[ComputeDistance] Computing distances for vector probes");
            std::size_t base_skip = 0;
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                auto const a_probe = probe_data->getProbe<probe::Vec4Probe>(a_pidx);
                auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                sample_count = a_samples_tmp.size();
                auto first_not_nan = std::find_if_not(a_samples_tmp.begin(), a_samples_tmp.end(), [](auto const& el) {
                    return std::isnan(el[0]) || std::isnan(el[1]) || std::isnan(el[2]) /*|| std::isnan(el[3])*/;
                });
                if (first_not_nan != a_samples_tmp.end()) {
                    auto first_not_nan_idx = std::distance(a_samples_tmp.begin(), first_not_nan);
                    if (first_not_nan_idx > base_skip) {
                        base_skip = first_not_nan_idx;
                    }
                }
            }
            if (base_skip > sample_count)
                base_skip = 0;
            core::utility::log::Log::DefaultLog.WriteInfo(
                "[ComputeDistance] Skipping first %d samples due to NaNs", base_skip);

            sample_count -= base_skip;

            auto min_val = std::numeric_limits<float>::max();
            auto max_val = std::numeric_limits<float>::lowest();


            auto vec_dist_func = [](glm::vec4 const& a, glm::vec4 const& b) -> float {
                auto const angle = std::acos(glm::dot(glm::vec3(a), glm::vec3(b)));
                auto const angle_dis = angle / M_PI;
                auto const length_dis = std::fabs(a.w - b.w);
                return angle_dis; /* + length_dis;*/
            };


            std::vector<std::vector<glm::vec4>> sample_collection(probe_count);
            auto X = std::vector<Eigen::MatrixXd>(probe_count, Eigen::MatrixXd(probe_count, sample_count));
#pragma omp parallel for
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                std::vector<glm::vec4> a_samples;
                {
                    auto const a_probe = probe_data->getProbe<probe::Vec4Probe>(a_pidx);
                    auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                    a_samples.resize(a_samples_tmp.size() - base_skip);
                    std::transform(a_samples_tmp.begin() + base_skip, a_samples_tmp.end(), a_samples.begin(),
                        [](auto const& el) { return glm::vec4(el[0], el[1], el[2], el[3]); });
                    std::for_each(a_samples.begin(), a_samples.end(), [](auto& el) {
                        auto const norm = glm::normalize(glm::vec3(el));
                        el.x = norm.x;
                        el.y = norm.y;
                        el.z = norm.z;
                    });
                }
                sample_collection[a_pidx] = a_samples;
            }
#pragma omp parallel for
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                auto& X_sub = X[a_pidx];
                for (std::int64_t b_pidx = 0; b_pidx < probe_count; ++b_pidx) {

                    for (std::int64_t as_idx = 0; as_idx < sample_count; ++as_idx) {
                        auto const val =
                            vec_dist_func(sample_collection[a_pidx][as_idx], sample_collection[b_pidx][as_idx]);
                        X_sub(b_pidx, as_idx) = val;
                    }
                }
                /*auto svd = Eigen::JacobiSVD<Eigen::MatrixXd>(X_sub, Eigen::ComputeThinU | Eigen::ComputeThinV);
                auto sv = svd.singularValues();
                for (Eigen::Index idx = sv.size() / 2; idx < sv.size(); ++idx) {
                    sv[idx] = 0.0;
                }
                auto U = svd.matrixU();
                auto V = svd.matrixV();
                auto D = sv.asDiagonal();
                X_sub = U * D * V.transpose();*/
            }

            core::utility::log::Log::DefaultLog.WriteInfo("[ComputeDistance] Prepared probes");
#pragma omp parallel for
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {

                for (std::int64_t b_pidx = a_pidx; b_pidx < probe_count; ++b_pidx) {


                    std::vector<float> tmp_mat(sample_count * sample_count);
                    for (std::int64_t as_idx = 0; as_idx < sample_count; ++as_idx) {
                        for (std::int64_t bs_idx = as_idx; bs_idx < sample_count; ++bs_idx) {
                            /*auto const val =
                                vec_dist_func(sample_collection[a_pidx][as_idx], sample_collection[b_pidx][bs_idx]);*/
                            //auto const val = std::abs(X[a_pidx](b_pidx, as_idx) - X[a_pidx](b_pidx, bs_idx));
                            auto const val = X[a_pidx](b_pidx, as_idx);
                            tmp_mat[as_idx + bs_idx * sample_count] = val;
                            tmp_mat[bs_idx + as_idx * sample_count] = val;
                        }
                    }


                    auto const score = datatools::misc::frechet_distance<float>(
                        sample_count, [&tmp_mat, sample_count](std::size_t lhs, std::size_t rhs) -> float {
                            return tmp_mat[lhs + rhs * sample_count];
                        });


                    _dis_mat[a_pidx + b_pidx * probe_count] = score;
                    _dis_mat[b_pidx + a_pidx * probe_count] = score;
#pragma omp critical
                    {
                        if (score < min_val)
                            min_val = score;
                        if (score > max_val)
                            max_val = score;
                    }
                }
            }
            /*if (min_val > 0.0)
                min_val = 0.0;*/
            auto org = min_val;
            auto diff = 1.0 / (max_val - min_val + 1e-8);
            double stretching = _stretching_factor_slot.Param<core::param::FloatParam>()->Value();
            /*std::for_each(_dis_mat.begin(), _dis_mat.end(), [org, diff](auto& el) {
                if (std::isnan(el))
                    el = 1.0;
            });*/
            std::for_each(_dis_mat.begin(), _dis_mat.end(), [org, diff](auto& el) { el = (el - org) * diff; });
            if (stretching > 1.0f) {
                std::for_each(_dis_mat.begin(), _dis_mat.end(), [stretching](auto& el) { el = el * stretching; });
                std::for_each(_dis_mat.begin(), _dis_mat.end(), [](auto& el) {
                    if (el > 1.0)
                        el = 1.0;
                });
            }
            core::utility::log::Log::DefaultLog.WriteInfo("[ComputeDistance] Finished");
        } else if (distrib_probe) {
            core::utility::log::Log::DefaultLog.WriteInfo(
                "[ComputeDistance] Computing distances for distribution probes");
            std::size_t base_skip = 0;
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                auto const a_probe = probe_data->getProbe<probe::FloatDistributionProbe>(a_pidx);
                auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                sample_count = a_samples_tmp.size();
                decltype(a_samples_tmp.begin()) first_not_nan = a_samples_tmp.end();
                for (auto it = a_samples_tmp.begin(); it != a_samples_tmp.end(); ++it) {
                    if (std::isnan<float>(it->mean)) {
                        first_not_nan = it;
                        break;
                    }
                }
                if (first_not_nan != a_samples_tmp.end()) {
                    auto first_not_nan_idx = std::distance(a_samples_tmp.begin(), first_not_nan);
                    if (first_not_nan_idx > base_skip) {
                        base_skip = first_not_nan_idx;
                    }
                }
            }
            if (base_skip > sample_count)
                base_skip = 0;
            core::utility::log::Log::DefaultLog.WriteInfo(
                "[ComputeDistance] Skipping first %d samples due to NaNs", base_skip);
            auto base_sample_count = sample_count - base_skip;
            auto X = Eigen::MatrixXd(probe_count, base_sample_count);
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                auto const a_probe = probe_data->getProbe<probe::FloatDistributionProbe>(a_pidx);
                auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                for (std::size_t sample_idx = base_skip; sample_idx < base_sample_count; ++sample_idx) {
                    X(a_pidx, sample_idx - base_skip) = a_samples_tmp[sample_idx].mean;
                }
            }
            /*auto svd = Eigen::JacobiSVD<Eigen::MatrixXd>(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
            auto sv = svd.singularValues();
            for (Eigen::Index idx = sv.size() / 2; idx < sv.size(); ++idx) {
                sv[idx] = 0.0;
            }
            auto U = svd.matrixU();
            auto V = svd.matrixV();
            auto D = sv.asDiagonal();
            X = U * D * V.transpose();*/

            auto min_val = std::numeric_limits<double>::max();
            auto max_val = std::numeric_limits<double>::lowest();
#pragma omp parallel for
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {

                for (std::int64_t b_pidx = a_pidx; b_pidx < probe_count; ++b_pidx) {


                    std::vector<float> tmp_mat(sample_count * sample_count);
                    for (std::int64_t as_idx = 0; as_idx < sample_count; ++as_idx) {
                        for (std::int64_t bs_idx = as_idx; bs_idx < sample_count; ++bs_idx) {
                            auto const val = std::abs(X(a_pidx, as_idx) - X(b_pidx, as_idx));
                            tmp_mat[as_idx + bs_idx * sample_count] = val;
                            tmp_mat[bs_idx + as_idx * sample_count] = val;
                        }
                    }

                    auto const score = datatools::misc::frechet_distance<float>(
                        sample_count, [&tmp_mat, sample_count](std::size_t lhs, std::size_t rhs) -> float {
                            return tmp_mat[lhs + rhs * sample_count];
                        });

                    // auto const dis = DTW::dtw_distance_only(a_samples, b_samples, 2);
                    _dis_mat[a_pidx + b_pidx * probe_count] = score;
                    _dis_mat[b_pidx + a_pidx * probe_count] = score;
#pragma omp critical
                    {
                        if (score < min_val)
                            min_val = score;
                        if (score > max_val)
                            max_val = score;
                    }
                }
            }
            auto org = min_val;
            auto diff = 1.0 / (max_val - min_val + 1e-8);
            double stretching = _stretching_factor_slot.Param<core::param::FloatParam>()->Value();
            std::for_each(_dis_mat.begin(), _dis_mat.end(), [org, diff](auto& el) { el = (el - org) * diff; });
            if (stretching > 1.0f) {
                std::for_each(_dis_mat.begin(), _dis_mat.end(), [stretching](auto& el) { el = el * stretching; });
                std::for_each(_dis_mat.begin(), _dis_mat.end(), [](auto& el) {
                    if (el > 1.0)
                        el = 1.0;
                });
            }
            core::utility::log::Log::DefaultLog.WriteInfo("[ComputeDistance] Finished");
        } else {
            core::utility::log::Log::DefaultLog.WriteInfo("[ComputeDistance] Computing distances for scalar probes");
            std::size_t base_skip = 0;
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                auto const a_probe = probe_data->getProbe<probe::FloatProbe>(a_pidx);
                auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                sample_count = a_samples_tmp.size();
                auto first_not_nan = std::find_if_not(a_samples_tmp.begin(), a_samples_tmp.end(), std::isnan<float>);
                if (first_not_nan != a_samples_tmp.end()) {
                    auto first_not_nan_idx = std::distance(a_samples_tmp.begin(), first_not_nan);
                    if (first_not_nan_idx > base_skip) {
                        base_skip = first_not_nan_idx;
                    }
                }
            }
            if (base_skip > sample_count)
                base_skip = 0;
            core::utility::log::Log::DefaultLog.WriteInfo(
                "[ComputeDistance] Skipping first %d samples due to NaNs", base_skip);

            // user defined range
            auto const bottom = _min_sample_slot.Param<core::param::IntParam>()->Value();
            auto top = _max_sample_slot.Param<core::param::IntParam>()->Value();
            std::size_t base_sample_count = 0;
            if (bottom >= 0 && top >= 1) {
                base_sample_count = top - bottom;
                base_skip = bottom;
            } else {
                base_sample_count = sample_count - base_skip;
                top = base_sample_count;
            }

            auto X = Eigen::MatrixXd(probe_count, base_sample_count);
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                auto const a_probe = probe_data->getProbe<probe::FloatProbe>(a_pidx);
                auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                for (std::size_t sample_idx = base_skip; sample_idx < top; ++sample_idx) {
                    X(a_pidx, sample_idx - base_skip) = a_samples_tmp[sample_idx];
                }
            }
            /*auto svd = Eigen::JacobiSVD<Eigen::MatrixXd>(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
            auto sv = svd.singularValues();
            for (Eigen::Index idx = sv.size() / 2; idx < sv.size(); ++idx) {
                sv[idx] = 0.0;
            }
            auto U = svd.matrixU();
            auto V = svd.matrixV();
            auto D = sv.asDiagonal();
            X = U * D * V.transpose();*/

            auto min_val = std::numeric_limits<double>::max();
            auto max_val = std::numeric_limits<double>::lowest();
#pragma omp parallel for
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {

                for (std::int64_t b_pidx = a_pidx; b_pidx < probe_count; ++b_pidx) {


                    std::vector<float> tmp_mat(base_sample_count * base_sample_count);
                    for (std::int64_t as_idx = 0; as_idx < base_sample_count; ++as_idx) {
                        for (std::int64_t bs_idx = as_idx; bs_idx < base_sample_count; ++bs_idx) {
                            auto const val = std::abs(X(a_pidx, as_idx) - X(b_pidx, as_idx));
                            tmp_mat[as_idx + bs_idx * base_sample_count] = val;
                            tmp_mat[bs_idx + as_idx * base_sample_count] = val;
                        }
                    }

                    auto const score = datatools::misc::frechet_distance<float>(
                        base_sample_count, [&tmp_mat, base_sample_count](std::size_t lhs, std::size_t rhs) -> float {
                            return tmp_mat[lhs + rhs * base_sample_count];
                        });

                    // auto const dis = DTW::dtw_distance_only(a_samples, b_samples, 2);
                    _dis_mat[a_pidx + b_pidx * probe_count] = score;
                    _dis_mat[b_pidx + a_pidx * probe_count] = score;
#pragma omp critical
                    {
                        if (score < min_val)
                            min_val = score;
                        if (score > max_val)
                            max_val = score;
                    }
                }
            }
            auto org = min_val;
            auto diff = 1.0 / (max_val - min_val + 1e-8);
            double stretching = _stretching_factor_slot.Param<core::param::FloatParam>()->Value();
            std::for_each(_dis_mat.begin(), _dis_mat.end(), [org, diff](auto& el) { el = (el - org) * diff; });
            if (stretching > 1.0f) {
                std::for_each(_dis_mat.begin(), _dis_mat.end(), [stretching](auto& el) { el = el * stretching; });
                std::for_each(_dis_mat.begin(), _dis_mat.end(), [](auto& el) {
                    if (el > 1.0)
                        el = 1.0;
                });
            }
            core::utility::log::Log::DefaultLog.WriteInfo(
                "[ComputeDistance] Finished with %f and %f", min_val, max_val);
        }
        for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
            auto const minmax = std::minmax_element(
                _dis_mat.begin() + (a_pidx * probe_count), _dis_mat.begin() + (a_pidx * probe_count + probe_count));
            _col_infos[a_pidx].SetName("p" + std::to_string(a_pidx));
            _col_infos[a_pidx].SetType(datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
            _col_infos[a_pidx].SetMinimumValue(*minmax.first);
            _col_infos[a_pidx].SetMaximumValue(*minmax.second);
        }

        _frame_id = meta_data.m_frame_ID;
        ++_out_data_hash;
        _stretching_factor_slot.ResetDirty();
    }

    out_table->SetFrameCount(meta_data.m_frame_cnt);
    out_table->SetFrameID(_frame_id);
    out_table->SetDataHash(_out_data_hash);
    out_table->Set(_col_count, _row_count, _col_infos.data(), _dis_mat.data());

    return true;
}


bool megamol::probe_gl::ComputeDistance::get_extent_cb(core::Call& c) {
    auto ctd = dynamic_cast<datatools::table::TableDataCall*>(&c);
    if (ctd == nullptr)
        return false;

    auto cpd = _in_probes_slot.CallAs<probe::CallProbes>();
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

bool megamol::probe_gl::ComputeDistance::paramChanged(core::param::ParamSlot& p) {

    _trigger_recalc = true;
    return true;
}
