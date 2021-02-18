#include "stdafx.h"
#include "ComputeDistance.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/CoreInstance.h"

#include "DTW.hpp"

#include "glm/glm.hpp"

#include "mmstd_datatools/misc/FrechetDistance.h"
//#include "mmstd_datatools/misc/KDE.h"

#include "eigen.h"

//#include "frechet_distance.h"


megamol::probe::ComputeDistance::ComputeDistance()
        : _out_table_slot("outTable", "")
        , _in_probes_slot("inProbes", "")
        , _stretching_factor_slot("stretching factor", "") {
    _out_table_slot.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(0), &ComputeDistance::get_data_cb);
    _out_table_slot.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(1), &ComputeDistance::get_extent_cb);
    MakeSlotAvailable(&_out_table_slot);

    _in_probes_slot.SetCompatibleCall<CallProbesDescription>();
    MakeSlotAvailable(&_in_probes_slot);

    _stretching_factor_slot << new core::param::FloatParam(5.0f, 0.0f);
    MakeSlotAvailable(&_stretching_factor_slot);
}


megamol::probe::ComputeDistance::~ComputeDistance() {
    this->Release();
}


bool megamol::probe::ComputeDistance::create() {
    /*try {
    if (!instance()->ShaderSourceFactory().MakeShaderSource("FrechetDistance::compute", _compute_shader_src))
        return false;
    vislib::graphics::gl::ShaderSource compute_shader_src = _compute_shader_src;
    std::string sample_count_decl = "const uint sample_count = 30;";
    vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet> snip = new vislib::graphics::gl::ShaderSource::StringSnippet(sample_count_decl.c_str());
    compute_shader_src.Insert(1, snip);

    if (!_fd_shader.Compile(compute_shader_src.Code(), compute_shader_src.Count()))
        return false;
    if (!_fd_shader.Link())
        return false;
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to compile shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader: Unknown exception\n");
        return false;
    }*/
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

    if (in_probes->hasUpdate() || meta_data.m_frame_ID != _frame_id || _stretching_factor_slot.IsDirty()) {
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
        std::size_t sample_count = 0;
        bool vec_probe = false;
        {
            auto test_probe = probe_data->getGenericProbe(0);
            vec_probe = std::holds_alternative<Vec4Probe>(test_probe);
        }

        if (vec_probe) {
            core::utility::log::Log::DefaultLog.WriteInfo("[ComputeDistance] Computing distances for vector probes");
            std::size_t base_skip = 0;
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                auto const a_probe = probe_data->getProbe<Vec4Probe>(a_pidx);
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
            /*std::vector<std::vector<double>> sample_collection(probe_count * probe_count);
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                std::vector<glm::vec3> a_samples;
                {
                    auto const a_probe = probe_data->getProbe<Vec4Probe>(a_pidx);
                    auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                    a_samples.resize(a_samples_tmp.size() - base_skip);
                    std::transform(a_samples_tmp.begin() + base_skip, a_samples_tmp.end(), a_samples.begin(),
                        [](auto const& el) { return glm::vec3(el[0], el[1], el[2]); });
                    std::for_each(a_samples.begin(), a_samples.end(), [](auto& el) { el = glm::normalize(el); });
                }
                for (std::int64_t b_pidx = a_pidx; b_pidx < probe_count; ++b_pidx) {
                    std::vector<glm::vec3> b_samples;
                    {
                        auto const b_probe = probe_data->getProbe<Vec4Probe>(b_pidx);
                        auto const& b_samples_tmp = b_probe.getSamplingResult()->samples;
                        b_samples.resize(b_samples_tmp.size() - base_skip);
                        std::transform(b_samples_tmp.begin() + base_skip, b_samples_tmp.end(), b_samples.begin(),
                            [](auto const& el) { return glm::vec3(el[0], el[1], el[2]); });
                        std::for_each(b_samples.begin(), b_samples.end(), [](auto& el) { el = glm::normalize(el); });
                    }
                    std::vector<double> current_samples(sample_count - base_skip);
                    std::transform(a_samples.begin(), a_samples.end(), b_samples.begin(), current_samples.begin(),
                        [](auto const& lhs, auto const& rhs) {
                            auto const angle = std::acos(glm::dot(lhs, rhs));
                            auto const angle_dis = angle / 3.14f;
                            return angle_dis;
                        });
                    sample_collection[a_pidx + b_pidx * probe_count] = current_samples;
                    sample_collection[b_pidx + a_pidx * probe_count] = current_samples;
                }
            }*/

            auto vec_dist_func = [](glm::vec3 const& a, glm::vec3 const& b) -> float {
                auto const angle = std::acos(glm::dot(a, b));
                auto const angle_dis = angle / 3.14f;
                return angle_dis;
            };

            /*std::vector<glm::vec3> sample_collection(probe_count * sample_count);
#pragma omp parallel for
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                std::vector<glm::vec3> a_samples;
                {
                    auto const a_probe = probe_data->getProbe<Vec4Probe>(a_pidx);
                    auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                    a_samples.resize(a_samples_tmp.size() - base_skip);
                    std::transform(a_samples_tmp.begin() + base_skip, a_samples_tmp.end(), a_samples.begin(),
                        [](auto const& el) { return glm::vec3(el[0], el[1], el[2]); });
                    std::for_each(a_samples.begin(), a_samples.end(), [](auto& el) { el = glm::normalize(el); });
                }
                for (unsigned int s_idx = 0; s_idx < sample_count; ++s_idx) {
                    sample_collection[a_pidx * sample_count + s_idx] = a_samples[s_idx];
                }
            }
            core::utility::log::Log::DefaultLog.WriteInfo("[ComputeDistance] Prepared probes");

            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);

            float3* samples = nullptr;
            cudaMalloc(&samples, probe_count * sample_count * sizeof(float3));
            auto error = cudaGetLastError();
            cudaMemcpy(
                samples, sample_collection.data(), sizeof(float3) * probe_count * sample_count, cudaMemcpyHostToDevice);
            error = cudaGetLastError();
            float* scores = nullptr;
            cudaMalloc(&scores, probe_count * probe_count * sizeof(float));
            error = cudaGetLastError();

            dim3 thread_dim(8, 8, 1);
            dim3 grid_dim(probe_count / 8 + 1, probe_count / 8 + 1, 1);

            launch_frechet_distance(grid_dim, thread_dim, scores, samples, probe_count, sample_count);
            error = cudaGetLastError();

            cudaDeviceSynchronize();
            error = cudaGetLastError();

            cudaMemcpy(_dis_mat.data(), scores, sizeof(float) * probe_count * probe_count, cudaMemcpyDeviceToHost);
            error = cudaGetLastError();*/

            std::vector<std::vector<glm::vec3>> sample_collection(probe_count);
#pragma omp parallel for
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                std::vector<glm::vec3> a_samples;
                {
                    auto const a_probe = probe_data->getProbe<Vec4Probe>(a_pidx);
                    auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                    a_samples.resize(a_samples_tmp.size() - base_skip);
                    std::transform(a_samples_tmp.begin() + base_skip, a_samples_tmp.end(), a_samples.begin(),
                        [](auto const& el) { return glm::vec3(el[0], el[1], el[2]); });
                    std::for_each(a_samples.begin(), a_samples.end(), [](auto& el) { el = glm::normalize(el); });
                }
                sample_collection[a_pidx] = a_samples;
            }
            core::utility::log::Log::DefaultLog.WriteInfo("[ComputeDistance] Prepared probes");
#pragma omp parallel for
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                /*std::vector<glm::vec3> a_samples;
                {
                    auto const a_probe = probe_data->getProbe<Vec4Probe>(a_pidx);
                    auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                    a_samples.resize(a_samples_tmp.size() - base_skip);
                    std::transform(a_samples_tmp.begin() + base_skip, a_samples_tmp.end(), a_samples.begin(),
                        [](auto const& el) { return glm::vec3(el[0], el[1], el[2]); });
                    std::for_each(a_samples.begin(), a_samples.end(), [](auto& el) { el = glm::normalize(el); });
                }*/
                for (std::int64_t b_pidx = a_pidx; b_pidx < probe_count; ++b_pidx) {
                    /*std::vector<glm::vec3> b_samples;
                    {
                        auto const b_probe = probe_data->getProbe<Vec4Probe>(b_pidx);
                        auto const& b_samples_tmp = b_probe.getSamplingResult()->samples;
                        b_samples.resize(b_samples_tmp.size() - base_skip);
                        std::transform(b_samples_tmp.begin() + base_skip, b_samples_tmp.end(), b_samples.begin(),
                            [](auto const& el) { return glm::vec3(el[0], el[1], el[2]); });
                        std::for_each(b_samples.begin(), b_samples.end(), [](auto& el) { el = glm::normalize(el); });
                    }*/
                    // std::vector<float> scores(a_samples.size());
                    // std::transform(a_samples.begin(), a_samples.end(), b_samples.begin(), scores.begin(),
                    //    [](auto const& lhs, auto const& rhs) {
                    //        if (std::isnan(lhs.x) || std::isnan(lhs.y) || std::isnan(lhs.z) ||
                    //            std::isnan(rhs.x) || std::isnan(rhs.y) || std::isnan(rhs.z)) {
                    //            return 1.0f;
                    //        }
                    //        // return glm::dot(lhs, rhs);
                    //        auto const angle = std::acos(glm::dot(lhs, rhs));
                    //        auto const angle_dis = angle / 3.14f;
                    //        return angle_dis;
                    //    });
                    // auto const score = std::accumulate(scores.begin(), scores.end(), 0.0f, std::plus<float>());
                    // auto const score = *std::max_element(scores.begin(), scores.end());
                    
                    std::vector<float> tmp_mat(sample_count * sample_count);
                    for (std::int64_t as_idx = 0; as_idx < sample_count; ++as_idx) {
                        for (std::int64_t bs_idx = as_idx; bs_idx < sample_count; ++bs_idx) {
                            auto const val = vec_dist_func(sample_collection[a_pidx][as_idx], sample_collection[b_pidx][bs_idx]);
                            tmp_mat[as_idx + bs_idx * sample_count] = val;
                            tmp_mat[bs_idx + as_idx * sample_count] = val;
                        }
                    }
                    /*auto const validation = *std::min_element(tmp_mat.begin(), tmp_mat.end(), [](auto lhs, auto rhs) {
                        if (lhs <= 0.0f)
                            return false;
                        return lhs < rhs;
                    });*/

                    /*auto validation = std::numeric_limits<float>::lowest();
                    for (std::int64_t as_idx = 0; as_idx < sample_count; ++as_idx) {
                        auto max_val = std::numeric_limits<float>::max(); 
                        for (std::int64_t bs_idx = 0; bs_idx < sample_count; ++bs_idx) {
                            auto const val = tmp_mat[as_idx + bs_idx * sample_count];
                            if (max_val > val) {
                                max_val = val;
                            }
                        }
                        if (validation < max_val) {
                            validation = max_val;
                        }
                    }*/
                    

                    auto const score = stdplugin::datatools::misc::frechet_distance<float>(
                        sample_count, [&tmp_mat, sample_count](std::size_t lhs, std::size_t rhs) -> float {
                            return tmp_mat[lhs + rhs * sample_count];
                        });

                    /*auto const validation = stdplugin::datatools::misc::frechet_distance_2<float>(
                        sample_count, [&tmp_mat, sample_count](std::size_t lhs, std::size_t rhs) -> float {
                            return tmp_mat[lhs + rhs * sample_count];
                        });

                    if (validation != score) {
                        core::utility::log::Log::DefaultLog.WriteInfo("score %f validation %f", score, validation);
                    }*/

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
            /*std::for_each(_dis_mat.begin(), _dis_mat.end(), [stretching](auto& el) { el = el * stretching; });
            std::for_each(_dis_mat.begin(), _dis_mat.end(), [](auto& el) {
                if (el > 1.0)
                    el = 1.0;
            });*/
            core::utility::log::Log::DefaultLog.WriteInfo("[ComputeDistance] Finished");
        } else {
            core::utility::log::Log::DefaultLog.WriteInfo("[ComputeDistance] Computing distances for scalar probes");
            std::size_t base_skip = 0;
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                auto const a_probe = probe_data->getProbe<FloatProbe>(a_pidx);
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
            auto base_sample_count = sample_count - base_skip;
            auto X = Eigen::MatrixXd(probe_count, base_sample_count);
            for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
                auto const a_probe = probe_data->getProbe<FloatProbe>(a_pidx);
                auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                for (std::size_t sample_idx = base_skip; sample_idx < base_sample_count; ++sample_idx) {
                    X(a_pidx, sample_idx - base_skip) = a_samples_tmp[sample_idx];
                }
            }
            auto svd = Eigen::JacobiSVD<Eigen::MatrixXd>(X, Eigen::ComputeFullU | Eigen::ComputeFullV);
            auto sv = svd.singularValues();
            for (Eigen::Index idx = sv.size() / 2; idx < sv.size(); ++idx) {
                sv[idx] = 0.0;
            }
            auto U = svd.matrixU();
            auto V = svd.matrixV();
            X = U * sv.asDiagonal() * V.transpose();

            auto min_val = std::numeric_limits<double>::max();
            auto max_val = std::numeric_limits<double>::lowest();
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
                a_samples.resize(X.cols());
                for (Eigen::Index idx = 0; idx < X.cols(); ++idx) {
                    a_samples[idx] = X(a_pidx, idx);
                }
                //{
                //    auto const a_probe = probe_data->getProbe<FloatProbe>(a_pidx);
                //    auto const& a_samples_tmp = a_probe.getSamplingResult()->samples;
                //    a_samples.resize(a_samples_tmp.size());
                //    std::transform(
                //        a_samples_tmp.cbegin(), a_samples_tmp.cend(), a_samples.begin(), [](auto const val) {
                //            return static_cast<double>(val);
                //        });
                //    /*auto const it = std::stable_partition(
                //        a_samples.begin(), a_samples.end(), [](auto const& el) { return !std::isnan(el); });
                //    std::for_each(it, a_samples.end(), [](auto& el) { el = 0.0; });*/
                //    /*auto first_val = std::find_if_not(a_samples.begin(), a_samples.end(), std::isnan<double>);
                //    auto last_val = std::find_if_not(a_samples.rbegin(), a_samples.rend(), std::isnan<double>);
                //    std::for_each(a_samples.begin(), first_val, [&first_val](auto& el){
                //        el = *first_val;});
                //    std::for_each(a_samples.rbegin(), last_val, [&last_val](auto& el) { el = *last_val; });
                //    auto a_tmp = a_samples;
                //    for (size_t idx = 0; idx < a_samples.size(); ++idx) {
                //        auto a = idx - 1;
                //        if (idx == 0)
                //            a = 0;
                //        auto b = idx;
                //        auto c = idx + 1;
                //        if (idx == a_samples.size() - 1)
                //            c = a_samples.size() - 1;

                //        a_samples[idx] = 0.5 * (0.5 * a_tmp[a] + a_tmp[b] + 0.5 * a_tmp[c]);
                //    }*/
                //    a_samples.erase(a_samples.begin(), a_samples.begin() + base_skip);
                //}
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
                    b_samples.resize(X.cols());
                    for (Eigen::Index idx = 0; idx < X.cols(); ++idx) {
                        b_samples[idx] = X(b_pidx, idx);
                    }
                    //{
                    //    auto const b_probe = probe_data->getProbe<FloatProbe>(b_pidx);
                    //    auto const b_samples_tmp = b_probe.getSamplingResult()->samples;
                    //    b_samples.resize(b_samples_tmp.size());
                    //    std::transform(b_samples_tmp.cbegin(), b_samples_tmp.cend(), b_samples.begin(),
                    //        [](auto const val) {
                    //            return static_cast<double>(val);
                    //        });
                    //    /*auto const it = std::stable_partition(
                    //        b_samples.begin(), b_samples.end(), [](auto const& el) { return !std::isnan(el); });
                    //    std::for_each(it, b_samples.end(), [](auto& el) { el = 0.0; });*/
                    //    /*auto first_val = std::find_if_not(b_samples.begin(), b_samples.end(), std::isnan<double>);
                    //    auto last_val = std::find_if_not(b_samples.rbegin(), b_samples.rend(), std::isnan<double>);
                    //    std::for_each(b_samples.begin(), first_val, [&first_val](auto& el) { el = *first_val; });
                    //    std::for_each(b_samples.rbegin(), last_val, [&last_val](auto& el) { el = *last_val; });
                    //    auto b_tmp = b_samples;
                    //    for (size_t idx = 0; idx < b_samples.size(); ++idx) {
                    //        auto a = idx - 1;
                    //        if (idx == 0)
                    //            a = 0;
                    //        auto b = idx;
                    //        auto c = idx + 1;
                    //        if (idx == b_samples.size() - 1)
                    //            c = b_samples.size() - 1;

                    //        b_samples[idx] = 0.5 * (0.5 * b_tmp[a] + b_tmp[b] + 0.5 * b_tmp[c]);
                    //    }*/
                    //    b_samples.erase(b_samples.begin(), b_samples.begin() + base_skip);
                    //}
                    auto const dis = stdplugin::datatools::misc::frechet_distance<double, double>(a_samples, b_samples,
                        [](double const& a, double const& b) -> double { return std::abs(a - b); });
                    // auto const dis = DTW::dtw_distance_only(a_samples, b_samples, 2);
                    _dis_mat[a_pidx + b_pidx * probe_count] = dis;
                    _dis_mat[b_pidx + a_pidx * probe_count] = dis;
#pragma omp critical
                    {
                        if (dis < min_val)
                            min_val = dis;
                        if (dis > max_val)
                            max_val = dis;
                    }
                }
            }
            auto org = min_val;
            auto diff = 1.0 / (max_val - min_val + 1e-8);
            double stretching = _stretching_factor_slot.Param<core::param::FloatParam>()->Value();
            std::for_each(_dis_mat.begin(), _dis_mat.end(), [org, diff](auto& el) { el = (el - org) * diff; });
            std::for_each(_dis_mat.begin(), _dis_mat.end(), [stretching](auto& el) { el = el * stretching; });
            std::for_each(_dis_mat.begin(), _dis_mat.end(), [](auto& el) {
                if (el > 1.0)
                    el = 1.0;
            });
            core::utility::log::Log::DefaultLog.WriteInfo("[ComputeDistance] Finished");
        }

        for (std::int64_t a_pidx = 0; a_pidx < probe_count; ++a_pidx) {
            auto const minmax = std::minmax_element(
                _dis_mat.begin() + (a_pidx * probe_count), _dis_mat.begin() + (a_pidx * probe_count + probe_count));
            _col_infos[a_pidx].SetName("p" + std::to_string(a_pidx));
            _col_infos[a_pidx].SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
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
