#include "VelocityDistribution.h"

#include <filesystem>
#include <fstream>

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"

#include "glm/glm.hpp"


megamol::thermodyn::VelocityDistribution::VelocityDistribution()
        : out_dist_slot_("outDist", "")
        , out_stat_slot_("outStats", "")
        , out_part_slot_("outPart", "")
        , in_data_slot_("inData", "")
        , num_buckets_slot_("num buckets", "")
        , dump_histo_slot_("dump", "")
        , path_slot_("path", "")
        , mode_slot_("mode", "") {
    out_dist_slot_.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(0), &VelocityDistribution::get_data_cb);
    out_dist_slot_.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(1), &VelocityDistribution::get_extent_cb);
    MakeSlotAvailable(&out_dist_slot_);

    out_stat_slot_.SetCallback(
        CallStatsInfo::ClassName(), CallStatsInfo::FunctionName(0), &VelocityDistribution::get_stats_data_cb);
    out_stat_slot_.SetCallback(
        CallStatsInfo::ClassName(), CallStatsInfo::FunctionName(1), &VelocityDistribution::get_stats_extent_cb);
    MakeSlotAvailable(&out_stat_slot_);

    out_part_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &VelocityDistribution::get_parts_data_cb);
    out_part_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &VelocityDistribution::get_parts_extent_cb);
    MakeSlotAvailable(&out_part_slot_);

    in_data_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&in_data_slot_);

    num_buckets_slot_ << new core::param::IntParam(10, 1);
    MakeSlotAvailable(&num_buckets_slot_);

    dump_histo_slot_ << new core::param::ButtonParam();
    dump_histo_slot_.SetUpdateCallback(&VelocityDistribution::dump_histo);
    MakeSlotAvailable(&dump_histo_slot_);

    path_slot_ << new core::param::FilePathParam("./");
    MakeSlotAvailable(&path_slot_);

    using mode_ut = std::underlying_type_t<mode>;
    auto ep = new core::param::EnumParam(static_cast<mode_ut>(mode::icol));
    ep->SetTypePair(static_cast<mode_ut>(mode::icol), "icol");
    ep->SetTypePair(static_cast<mode_ut>(mode::dir), "dir");
    mode_slot_ << ep;
    MakeSlotAvailable(&mode_slot_);
}


megamol::thermodyn::VelocityDistribution::~VelocityDistribution() {
    this->Release();
}


bool megamol::thermodyn::VelocityDistribution::create() {
    return true;
}


void megamol::thermodyn::VelocityDistribution::release() {}


bool megamol::thermodyn::VelocityDistribution::assert_data(megamol::core::moldyn::MultiParticleDataCall& call) {
    auto const pl_count = call.GetParticleListCount();

    auto const num_buckets = num_buckets_slot_.Param<core::param::IntParam>()->Value();
    auto const mode_type = static_cast<mode>(mode_slot_.Param<core::param::EnumParam>()->Value());

    histograms_.clear();
    domain_.clear();

    part_data_.resize(pl_count);

    for (std::remove_const_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& part = call.AccessParticles(pl_idx);
        core::utility::log::Log::DefaultLog.WriteInfo("[VelocityDistribution] Computing stuff");
        /*auto& histo = histograms_[pl_idx];
        histo.clear();
        histo.resize(num_buckets, 0);
        auto& domain = domain_[pl_idx];
        domain.clear();
        domain.resize(num_buckets);*/

        auto const p_count = part.GetCount();

        auto& part_data = part_data_[pl_idx];
        part_data.clear();
        part_data.reserve(p_count * 7);

        switch (mode_type) {
        case mode::icol: {
            auto const type = part.GetColourDataType();
            if (type == part.COLDATA_FLOAT_I || type == part.COLDATA_DOUBLE_I) {
                auto const min_i = part.GetMinColourIndexValue();
                auto const max_i = part.GetMaxColourIndexValue();

                auto const fac_i = 1.0f / (max_i - min_i + 1e-8f);

                auto const diff_i = (max_i - min_i) / num_buckets;

                auto const xAcc = part.GetParticleStore().GetXAcc();
                auto const yAcc = part.GetParticleStore().GetYAcc();
                auto const zAcc = part.GetParticleStore().GetZAcc();

                auto const iAcc = part.GetParticleStore().GetCRAcc();

                decltype(histograms_)::value_type histo(num_buckets, 0);
                decltype(domain_)::value_type domain(num_buckets);

                std::generate(domain.begin(), domain.end(), [diff_i, val = min_i]() mutable {
                    auto old_val = val;
                    val += diff_i;
                    return old_val;
                });

                for (std::remove_const_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                    auto const val = iAcc->Get_f(p_idx);

                    auto const idx = static_cast<int>(((val - min_i) * fac_i) * static_cast<float>(num_buckets - 1));

                    ++histo[idx];

                    part_data.push_back(xAcc->Get_f(p_idx));
                    part_data.push_back(yAcc->Get_f(p_idx));
                    part_data.push_back(zAcc->Get_f(p_idx));
                    part_data.push_back(iAcc->Get_f(p_idx));
                    part_data.push_back(0.0f);
                    part_data.push_back(0.0f);
                    part_data.push_back(0.0f);
                }

                histograms_.push_back(histo);
                domain_.push_back(domain);
            }
        } break;
        case mode::dir: {
            auto const type = part.GetDirDataType();
            if (type != core::moldyn::SimpleSphericalParticles::DirDataType::DIRDATA_NONE) {
                auto const xAcc = part.GetParticleStore().GetXAcc();
                auto const yAcc = part.GetParticleStore().GetYAcc();
                auto const zAcc = part.GetParticleStore().GetZAcc();
                auto const dxAcc = part.GetParticleStore().GetDXAcc();
                auto const dyAcc = part.GetParticleStore().GetDYAcc();
                auto const dzAcc = part.GetParticleStore().GetDZAcc();

                std::vector<float> temp_dx(p_count);
                std::vector<float> temp_dy(p_count);
                std::vector<float> temp_dz(p_count);
                std::vector<float> temp_mag(p_count);

                for (std::remove_const_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                    auto const x_val = dxAcc->Get_f(p_idx);
                    auto const y_val = dyAcc->Get_f(p_idx);
                    auto const z_val = dzAcc->Get_f(p_idx);
                    temp_dx[p_idx] = x_val;
                    temp_dy[p_idx] = y_val;
                    temp_dz[p_idx] = z_val;
                    temp_mag[p_idx] = std::sqrtf(x_val * x_val + y_val * y_val + z_val * z_val);
                }

                auto const minmax_dx = std::minmax_element(temp_dx.begin(), temp_dx.end());
                auto const minmax_dy = std::minmax_element(temp_dy.begin(), temp_dy.end());
                auto const minmax_dz = std::minmax_element(temp_dz.begin(), temp_dz.end());
                auto const minmax_mag = std::minmax_element(temp_mag.begin(), temp_mag.end());

                decltype(histograms_)::value_type histo_dx(num_buckets, 0);
                decltype(domain_)::value_type domain_dx(num_buckets);
                decltype(histograms_)::value_type histo_dy(num_buckets, 0);
                decltype(domain_)::value_type domain_dy(num_buckets);
                decltype(histograms_)::value_type histo_dz(num_buckets, 0);
                decltype(domain_)::value_type domain_dz(num_buckets);
                decltype(histograms_)::value_type histo_mag(num_buckets, 0);
                decltype(domain_)::value_type domain_mag(num_buckets);

                auto const min_dx_i = *minmax_dx.first;
                auto const max_dx_i = *minmax_dx.second;

                auto const fac_dx_i = 1.0f / (max_dx_i - min_dx_i + 1e-8f);

                auto const diff_dx_i = (max_dx_i - min_dx_i) / num_buckets;

                auto const min_dy_i = *minmax_dy.first;
                auto const max_dy_i = *minmax_dy.second;

                auto const fac_dy_i = 1.0f / (max_dy_i - min_dy_i + 1e-8f);

                auto const diff_dy_i = (max_dy_i - min_dy_i) / num_buckets;

                auto const min_dz_i = *minmax_dz.first;
                auto const max_dz_i = *minmax_dz.second;

                auto const fac_dz_i = 1.0f / (max_dz_i - min_dz_i + 1e-8f);

                auto const diff_dz_i = (max_dz_i - min_dz_i) / num_buckets;

                auto const min_mag_i = *minmax_mag.first;
                auto const max_mag_i = *minmax_mag.second;

                auto const fac_mag_i = 1.0f / (max_mag_i - min_mag_i + 1e-8f);

                auto const diff_mag_i = (max_mag_i - min_mag_i) / num_buckets;

                std::generate(domain_dx.begin(), domain_dx.end(), [diff_dx_i, val = min_dx_i]() mutable {
                    auto old_val = val;
                    val += diff_dx_i;
                    return old_val;
                });
                std::generate(domain_dy.begin(), domain_dy.end(), [diff_dy_i, val = min_dy_i]() mutable {
                    auto old_val = val;
                    val += diff_dy_i;
                    return old_val;
                });
                std::generate(domain_dz.begin(), domain_dz.end(), [diff_dz_i, val = min_dz_i]() mutable {
                    auto old_val = val;
                    val += diff_dz_i;
                    return old_val;
                });
                std::generate(domain_mag.begin(), domain_mag.end(), [diff_mag_i, val = min_mag_i]() mutable {
                    auto old_val = val;
                    val += diff_mag_i;
                    return old_val;
                });

                for (std::remove_const_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                    auto const val_dx = temp_dx[p_idx];
                    auto const val_dy = temp_dy[p_idx];
                    auto const val_dz = temp_dz[p_idx];
                    auto const val_mag = temp_mag[p_idx];

                    auto const idx_dx =
                        static_cast<int>(((val_dx - min_dx_i) * fac_dx_i) * static_cast<float>(num_buckets - 1));
                    auto const idx_dy =
                        static_cast<int>(((val_dy - min_dy_i) * fac_dy_i) * static_cast<float>(num_buckets - 1));
                    auto const idx_dz =
                        static_cast<int>(((val_dz - min_dz_i) * fac_dz_i) * static_cast<float>(num_buckets - 1));
                    auto const idx_mag =
                        static_cast<int>(((val_mag - min_mag_i) * fac_mag_i) * static_cast<float>(num_buckets - 1));

                    ++histo_dx[idx_dx];
                    ++histo_dy[idx_dy];
                    ++histo_dz[idx_dz];
                    ++histo_mag[idx_mag];

                    part_data.push_back(xAcc->Get_f(p_idx));
                    part_data.push_back(yAcc->Get_f(p_idx));
                    part_data.push_back(zAcc->Get_f(p_idx));
                    part_data.push_back(val_mag);
                    part_data.push_back(val_dx);
                    part_data.push_back(val_dy);
                    part_data.push_back(val_dz);
                }

                histograms_.push_back(histo_dx);
                histograms_.push_back(histo_dy);
                histograms_.push_back(histo_dz);
                histograms_.push_back(histo_mag);

                domain_.push_back(domain_dx);
                domain_.push_back(domain_dy);
                domain_.push_back(domain_dz);
                domain_.push_back(domain_mag);
            }
        } break;
        }
    }
    compute_statistics();

    col_cnt_ = histograms_.size();
    row_cnt_ = num_buckets;

    ++out_data_hash_;

    return true;
}


bool megamol::thermodyn::VelocityDistribution::dump_histo(core::param::ParamSlot& p) {
    auto const path = std::filesystem::path(path_slot_.Param<core::param::FilePathParam>()->Value().PeekBuffer());

    // compute_statistics();

    for (std::size_t pl_idx = 0; pl_idx < histograms_.size(); ++pl_idx) {
        auto const& histo = histograms_[pl_idx];
        auto const& domain = domain_[pl_idx];

        auto const filepath = path / ("histo_" + std::to_string(pl_idx) + ".txt");

        auto ofile = std::ofstream(filepath, std::ios::out);

        for (auto const& el : domain) {
            ofile << std::to_string(el) << ",";
        }

        ofile.seekp(-1, std::ios::end);
        ofile << "\n";

        for (auto const& el : histo) {
            ofile << std::to_string(el) << ",";
        }

        ofile.seekp(-1, std::ios::end);
        ofile << "\n";
    }

    return true;
}


void megamol::thermodyn::VelocityDistribution::compute_statistics() {
    mean_.resize(histograms_.size());
    stddev_.resize(histograms_.size());

    for (std::size_t pl_idx = 0; pl_idx < histograms_.size(); ++pl_idx) {
        auto const& histo = histograms_[pl_idx];
        auto const& domain = domain_[pl_idx];

        std::vector<float> tmp_avg(histo.size());
        std::transform(histo.begin(), histo.end(), domain.begin(), tmp_avg.begin(),
            [](auto const& h, auto const& d) { return static_cast<float>(h) * d; });

        auto const sum_weights = std::accumulate(histo.begin(), histo.end(), 0.0f);
        auto const weighted_mean = std::accumulate(tmp_avg.begin(), tmp_avg.end(), 0.0f);
        auto const mean = weighted_mean / sum_weights;

        auto const fac = sum_weights * static_cast<float>((histo.size() - 1)) / static_cast<float>(histo.size());
        std::transform(histo.begin(), histo.end(), domain.begin(), tmp_avg.begin(),
            [mean](auto const& h, auto const& d) { return static_cast<float>(h) * (d - mean) * (d - mean); });
        auto const stddev = std::sqrtf(std::accumulate(tmp_avg.begin(), tmp_avg.end(), 0.0f) / fac);

        mean_[pl_idx] = mean;
        stddev_[pl_idx] = stddev;

        // core::utility::log::Log::DefaultLog.WriteInfo("[VelocityDistribution] Mean %f Stddev %f", mean, stddev);
    }
}


bool megamol::thermodyn::VelocityDistribution::get_data_cb(core::Call& c) {
    auto out_dist = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (out_dist == nullptr)
        return false;
    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;
    auto tmp_frame_id = out_dist->GetFrameID();
    in_data->SetFrameID(tmp_frame_id);
    if (!(*in_data)(1))
        return false;
    if (!(*in_data)(0))
        return false;

    if (in_data->DataHash() != in_data_hash_ || in_data->FrameID() != frame_id_ || is_dirty()) {
        auto const res = assert_data(*in_data);

        std::size_t allel = 0;
        for (std::size_t i = 0; i < histograms_.size(); ++i) {
            allel += histograms_[i].size();
        }

        data_.clear();
        data_.reserve(allel);
        ci_.clear();
        ci_.resize(histograms_.size());

        auto wh = data_.begin();
        for (std::size_t i = 0; i < histograms_.size(); ++i) {
            wh = data_.insert(wh, histograms_[i].begin(), histograms_[i].end());
            auto const minmax = std::minmax_element(histograms_[i].begin(), histograms_[i].end());
            ci_[i].SetMinimumValue(*minmax.first);
            ci_[i].SetMaximumValue(*minmax.second);
            ci_[i].SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
            ci_[i].SetName(std::string("pl_") + std::to_string(i));
        }

        reset_dirty();
        frame_id_ = tmp_frame_id;
        in_data_hash_ = in_data->DataHash();
    }

    out_dist->Set(col_cnt_, row_cnt_, ci_.data(), data_.data());
    out_dist->SetDataHash(out_data_hash_);
    out_dist->SetFrameID(frame_id_);

    return true;
}


bool megamol::thermodyn::VelocityDistribution::get_extent_cb(core::Call& c) {
    auto out_dist = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (out_dist == nullptr)
        return false;
    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    if (!(*in_data)(1))
        return false;

    out_dist->SetFrameCount(in_data->FrameCount());
    out_dist->SetDataHash(out_data_hash_);

    return true;
}


bool megamol::thermodyn::VelocityDistribution::get_stats_data_cb(core::Call& c) {
    auto out_stats = dynamic_cast<CallStatsInfo*>(&c);
    if (out_stats == nullptr)
        return false;

    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    // in_data->SetFrameID(out_dist->GetFrameID());
    auto meta = out_stats->getMetaData();

    in_data->SetFrameID(meta.m_frame_ID);
    if (!(*in_data)(1))
        return false;
    if (!(*in_data)(0))
        return false;

    if (in_data->DataHash() != in_data_hash_ || in_data->FrameID() != frame_id_ || is_dirty()) {
        auto const res = assert_data(*in_data);

        in_data_hash_ = in_data->DataHash();
        frame_id_ = in_data->FrameID();
        reset_dirty();
    }

    CallStatsInfo::data_type tmp_data(mean_.size());

    for (auto i = 0ull; i < tmp_data.size(); ++i) {
        tmp_data[i].name = std::string("d") + std::to_string(i);
        tmp_data[i].mean = mean_[i];
        tmp_data[i].stddev = stddev_[i];
    }

    out_stats->setData(tmp_data, out_data_hash_);
    // core::utility::log::Log::DefaultLog.WriteInfo("[VelocityDistribution] Setting stats data %d", tmp_data.size());

    meta.m_frame_ID = in_data->FrameID();

    out_stats->setMetaData(meta);

    return true;
}


bool megamol::thermodyn::VelocityDistribution::get_stats_extent_cb(core::Call& c) {
    auto out_stats = dynamic_cast<CallStatsInfo*>(&c);
    if (out_stats == nullptr)
        return false;

    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    auto meta = out_stats->getMetaData();

    in_data->SetFrameID(meta.m_frame_ID);
    if (!(*in_data)(1))
        return false;

    meta.m_frame_cnt = in_data->FrameCount();
    meta.m_frame_ID = in_data->FrameID();

    out_stats->setMetaData(meta);

    return true;
}


bool megamol::thermodyn::VelocityDistribution::get_parts_data_cb(core::Call& c) {
    auto out_data = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_data == nullptr)
        return false;

    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    in_data->SetFrameID(out_data->FrameID());
    if (!(*in_data)(1))
        return false;
    if (!(*in_data)(0))
        return false;

    if (in_data->DataHash() != in_data_hash_ || in_data->FrameID() != frame_id_ || is_dirty()) {
        auto const res = assert_data(*in_data);

        in_data_hash_ = in_data->DataHash();
        frame_id_ = in_data->FrameID();
        reset_dirty();
    }


    out_data->SetParticleListCount(in_data->GetParticleListCount());

    for (std::remove_const_t<decltype(in_data->GetParticleListCount())> pl_idx = 0;
         pl_idx < in_data->GetParticleListCount(); ++pl_idx) {

        auto const& part_data = part_data_[pl_idx];
        auto& parts = out_data->AccessParticles(pl_idx);

        parts.SetCount(part_data.size() / 7);
        parts.SetVertexData(
            core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, part_data.data(), 7 * sizeof(float));
        parts.SetColourData(
            core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, part_data.data() + 3, 7 * sizeof(float));
        parts.SetDirData(
            core::moldyn::SimpleSphericalParticles::DIRDATA_FLOAT_XYZ, part_data.data() + 4, 7 * sizeof(float));
        parts.SetGlobalRadius(in_data->AccessParticles(pl_idx).GetGlobalRadius());
        parts.SetColourMapIndexValues(in_data->AccessParticles(pl_idx).GetMinColourIndexValue(),
            in_data->AccessParticles(pl_idx).GetMaxColourIndexValue());
    }


    auto const bbox = in_data->AccessBoundingBoxes().ObjectSpaceBBox();
    auto const cbox = in_data->AccessBoundingBoxes().ObjectSpaceClipBox();

    out_data->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
    out_data->AccessBoundingBoxes().SetObjectSpaceClipBox(cbox);


    return true;
}


bool megamol::thermodyn::VelocityDistribution::get_parts_extent_cb(core::Call& c) {
    auto out_data = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_data == nullptr)
        return false;

    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    in_data->SetFrameID(out_data->FrameID());
    if (!(*in_data)(1))
        return false;

    auto const bbox = in_data->AccessBoundingBoxes().ObjectSpaceBBox();
    auto const cbox = in_data->AccessBoundingBoxes().ObjectSpaceClipBox();

    out_data->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
    out_data->AccessBoundingBoxes().SetObjectSpaceClipBox(cbox);

    out_data->SetFrameCount(in_data->FrameCount());
    out_data->SetFrameID(in_data->FrameID());

    return true;
}
