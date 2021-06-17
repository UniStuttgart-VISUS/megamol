#include "IColStatistics.h"

#include <algorithm>

#include "mmcore/param/IntParam.h"


megamol::stdplugin::datatools::IColStatistics::IColStatistics()
        : data_out_slot_("dataOut", ""), data_in_slot_("dataIn", ""), num_buckets_slot_("num buckets", "") {
    data_out_slot_.SetCallback(
        StatisticsCall::ClassName(), StatisticsCall::FunctionName(0), &IColStatistics::get_data_cb);
    data_out_slot_.SetCallback(
        StatisticsCall::ClassName(), StatisticsCall::FunctionName(1), &IColStatistics::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    num_buckets_slot_ << new core::param::IntParam(100, 1);
    MakeSlotAvailable(&num_buckets_slot_);
}


megamol::stdplugin::datatools::IColStatistics::~IColStatistics() {
    this->Release();
}


bool megamol::stdplugin::datatools::IColStatistics::create() {

    return true;
}


void megamol::stdplugin::datatools::IColStatistics::release() {}


bool megamol::stdplugin::datatools::IColStatistics::get_data_cb(core::Call& c) {
    auto data_out = dynamic_cast<StatisticsCall*>(&c);
    if (data_out == nullptr)
        return false;

    auto parts_in = data_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto meta = data_out->getMetaData();
    parts_in->SetFrameID(meta.m_frame_ID);
    if (!(*parts_in)(0))
        return false;

    if (parts_in->DataHash() != in_data_hash_ || parts_in->FrameID() != frame_id_) {
        if (!assert_data(*parts_in))
            return false;

        in_data_hash_ = parts_in->DataHash();
        frame_id_ = parts_in->FrameID();
        ++out_data_hash_;
    }

    meta = data_out->getMetaData();
    meta.m_frame_ID = frame_id_;
    data_out->setMetaData(meta);
    data_out->setData(data_, out_data_hash_);

    return true;
}


bool megamol::stdplugin::datatools::IColStatistics::get_extent_cb(core::Call& c) {
    auto data_out = dynamic_cast<StatisticsCall*>(&c);
    if (data_out == nullptr)
        return false;

    auto parts_in = data_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto meta = data_out->getMetaData();
    parts_in->SetFrameID(meta.m_frame_ID);
    if (!(*parts_in)(1))
        return false;

    meta.m_frame_cnt = parts_in->FrameCount();
    data_out->setMetaData(meta);

    return true;
}


bool megamol::stdplugin::datatools::IColStatistics::assert_data(core::moldyn::MultiParticleDataCall& in_parts) {
    auto const num_buckets = num_buckets_slot_.Param<core::param::IntParam>()->Value();

    auto const pl_count = in_parts.GetParticleListCount();

    data_.resize(pl_count);

    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& parts = in_parts.AccessParticles(pl_idx);
        auto& data = data_[pl_idx];

        auto const p_count = parts.GetCount();

        std::vector<float> icol_values;
        icol_values.reserve(p_count);

        for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
            auto const i_acc = parts.GetParticleStore().GetCRAcc();

            icol_values.push_back(i_acc->Get_f(p_idx));
        }

        auto const avg_val = std::accumulate(icol_values.begin(), icol_values.end(), 0.f) / static_cast<float>(p_count);
        std::nth_element(icol_values.begin(), icol_values.begin() + (p_count / 2), icol_values.end());
        auto const med_val = *(icol_values.begin() + (p_count / 2));
        auto const minmax_el = std::minmax_element(icol_values.begin(), icol_values.end());

        std::vector<float> histo(num_buckets);

        auto const min_val = *minmax_el.first;
        auto const max_val = *minmax_el.second;
        auto const diff = 1.f / ((max_val - min_val) + 1e-8f);

        for (auto const val : icol_values) {
            auto const idx =
                static_cast<uint32_t>(std::floorf((val - min_val) * diff * static_cast<float>(num_buckets)));
            if (idx >= 0 && idx < num_buckets)
                histo[idx] += 1.f;
        }

        auto const minmax_histo_el = std::minmax_element(histo.begin(), histo.end());

        auto const histo_max = *minmax_histo_el.second;

        std::for_each(histo.begin(), histo.end(), [histo_max](auto& val) { val /= histo_max; });

        data.avg_val = avg_val;
        data.med_val = med_val;
        data.min_val = min_val;
        data.max_val = max_val;
        data.histo = std::move(histo);
    }

    return true;
}
