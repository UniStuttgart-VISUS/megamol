#include "VelocityDistribution.h"

#include <filesystem>
#include <fstream>

#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"

#include "glm/glm.hpp"


megamol::thermodyn::VelocityDistribution::VelocityDistribution()
        : out_dist_slot_("outDist", "")
        , in_data_slot_("inData", "")
        , num_buckets_slot_("num buckets", "")
        , dump_histo_slot_("dump", "")
        , path_slot_("path", "")
        , mode_slot_("mode", "") {
    out_dist_slot_.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(0), &VelocityDistribution::get_data_cb);
    out_dist_slot_.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(1), &VelocityDistribution::get_extent_sb);
    MakeSlotAvailable(&out_dist_slot_);

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

    histograms_.resize(pl_count);

    col_cnt_ = pl_count;
    row_cnt_ = num_buckets;

    for (std::remove_const_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& part = call.AccessParticles(pl_idx);

        auto& histo = histograms_[pl_idx];
        histo.clear();
        histo.resize(num_buckets, 0);

        auto const p_count = part.GetCount();

        switch (mode_type) {
        case mode::icol: {
            auto const type = part.GetColourDataType();
            if (type == part.COLDATA_FLOAT_I || type == part.COLDATA_DOUBLE_I) {
                auto const min_i = part.GetMinColourIndexValue();
                auto const max_i = part.GetMaxColourIndexValue();

                auto const fac_i = 1.0f / (max_i - min_i + 1e-8f);

                auto const iAcc = part.GetParticleStore().GetCRAcc();

                for (std::remove_const_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                    auto const val = iAcc->Get_f(p_idx);

                    auto const idx = static_cast<int>(((val - min_i) * fac_i) * static_cast<float>(num_buckets - 1));

                    ++histo[idx];
                }
            }
        } break;
        case mode::dir: {
            auto const type = part.GetDirDataType();
            if (type != core::moldyn::SimpleSphericalParticles::DirDataType::DIRDATA_NONE) {
                auto const dxAcc = part.GetParticleStore().GetDXAcc();
                auto const dyAcc = part.GetParticleStore().GetDYAcc();
                auto const dzAcc = part.GetParticleStore().GetDZAcc();

                std::vector<float> temp_mag(p_count);

                for (std::remove_const_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                    auto const x_val = dxAcc->Get_f(p_idx);
                    auto const y_val = dyAcc->Get_f(p_idx);
                    auto const z_val = dzAcc->Get_f(p_idx);
                    temp_mag[p_idx] = std::sqrtf(x_val * x_val + y_val * y_val + z_val * z_val);
                }

                auto const minmax = std::minmax_element(temp_mag.begin(), temp_mag.end());

                auto const min_i = *minmax.first;
                auto const max_i = *minmax.second;

                auto const fac_i = 1.0f / (max_i - min_i + 1e-8f);

                for (std::remove_const_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                    auto const val = temp_mag[p_idx];

                    auto const idx = static_cast<int>(((val - min_i) * fac_i) * static_cast<float>(num_buckets - 1));

                    ++histo[idx];
                }
            }
        } break;
        }
    }

    return true;
}


bool megamol::thermodyn::VelocityDistribution::dump_histo(core::param::ParamSlot& p) {
    auto const path = std::filesystem::path(path_slot_.Param<core::param::FilePathParam>()->Value().PeekBuffer());

    for (std::size_t pl_idx = 0; pl_idx < histograms_.size(); ++pl_idx) {
        auto const& histo = histograms_[pl_idx];

        auto const filepath = path / ("histo_" + std::to_string(pl_idx) + ".txt");

        auto ofile = std::ofstream(filepath, std::ios::out);

        for (auto const& el : histo) {
            ofile << std::to_string(el) << ",";
        }

        ofile.seekp(-1, std::ios::end);
        ofile << "\n";
    }

    return true;
}


bool megamol::thermodyn::VelocityDistribution::get_data_cb(core::Call& c) {
    auto out_dist = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (out_dist == nullptr)
        return false;
    auto in_data = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    in_data->SetFrameID(out_dist->GetFrameID());
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
        frame_id_ = in_data->FrameID();
        in_data_hash_ = in_data->DataHash();
        ++out_data_hash_;
    }

    out_dist->Set(col_cnt_, row_cnt_, ci_.data(), data_.data());
    out_dist->SetDataHash(out_data_hash_);
    out_dist->SetFrameID(frame_id_);

    return true;
}


bool megamol::thermodyn::VelocityDistribution::get_extent_sb(core::Call& c) {
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
