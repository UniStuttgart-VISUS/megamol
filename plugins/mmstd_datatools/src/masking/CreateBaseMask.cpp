#include "CreateBaseMask.h"


megamol::stdplugin::datatools::masking::CreateBaseMask::CreateBaseMask()
        : data_out_slot_("dataOut", "")
        , offsets_out_slot_("offsetsOut", "")
        , data_in_slot_("dataIn", "")
        , flags_read_slot_("flagsRead", "")
        , flags_write_slot_("flagsWrite", "") {
    data_out_slot_.SetCallback<core::moldyn::MultiParticleDataCall, 0>(&CreateBaseMask::get_data_cb);
    data_out_slot_.SetCallback<core::moldyn::MultiParticleDataCall, 1>(&CreateBaseMask::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    offsets_out_slot_.SetCallback<CallMaskOffsets, 0>(&CreateBaseMask::get_offsets_cb);
    offsets_out_slot_.SetCallback<CallMaskOffsets, 1>(&CreateBaseMask::dummy_cb);
    MakeSlotAvailable(&offsets_out_slot_);

    data_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    flags_read_slot_.SetCompatibleCall<core::FlagCallRead_CPUDescription>();
    MakeSlotAvailable(&flags_read_slot_);

    flags_write_slot_.SetCompatibleCall<core::FlagCallWrite_CPUDescription>();
    MakeSlotAvailable(&flags_write_slot_);
}


megamol::stdplugin::datatools::masking::CreateBaseMask::~CreateBaseMask() {
    this->Release();
}


bool megamol::stdplugin::datatools::masking::CreateBaseMask::create() {
    inclusive_sum_ = std::make_shared<offsets_t>();
    return true;
}


void megamol::stdplugin::datatools::masking::CreateBaseMask::release() {}


bool megamol::stdplugin::datatools::masking::CreateBaseMask::get_data_cb(core::Call& c) {
    auto data_out = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (data_out == nullptr)
        return false;

    auto data_in = data_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (data_in == nullptr)
        return false;

    auto flags_read = flags_read_slot_.CallAs<core::FlagCallRead_CPU>();
    if (flags_read == nullptr)
        return false;

    auto flags_write = flags_write_slot_.CallAs<core::FlagCallWrite_CPU>();
    if (flags_write == nullptr)
        return false;

    data_in->SetFrameID(data_out->FrameID());
    if (!(*data_in)(0))
        return false;

    if (data_in->DataHash() != in_data_hash_ || data_in->FrameID() != frame_id_) {
        if (!assert_data(*data_in, *flags_read, *flags_write))
            return false;
        in_data_hash_ = data_in->DataHash();
        frame_id_ = data_in->FrameID();
        ++offsets_version_;
    }

    *data_out = *data_in;
    data_in->SetUnlocker(nullptr, false);

    return true;
}


bool megamol::stdplugin::datatools::masking::CreateBaseMask::get_extent_cb(core::Call& c) {
    auto data_out = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (data_out == nullptr)
        return false;

    auto data_in = data_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (data_in == nullptr)
        return false;

    auto flags_read = flags_read_slot_.CallAs<core::FlagCallRead_CPU>();
    if (flags_read == nullptr)
        return false;

    auto flags_write = flags_write_slot_.CallAs<core::FlagCallWrite_CPU>();
    if (flags_write == nullptr)
        return false;

    data_in->SetFrameID(data_out->FrameID());
    if (!(*data_in)(1))
        return false;

    data_out->SetFrameCount(data_in->FrameCount());
    data_out->AccessBoundingBoxes() = data_in->AccessBoundingBoxes();

    return true;
}


bool megamol::stdplugin::datatools::masking::CreateBaseMask::get_offsets_cb(core::Call& c) {
    auto data_out = dynamic_cast<CallMaskOffsets*>(&c);
    if (data_out == nullptr)
        return false;

    data_out->setData(inclusive_sum_, offsets_version_);

    return true;
}


bool megamol::stdplugin::datatools::masking::CreateBaseMask::assert_data(core::moldyn::MultiParticleDataCall& particles,
    core::FlagCallRead_CPU& flags_read, core::FlagCallWrite_CPU& flags_write) {
    auto const pl_count = particles.GetParticleListCount();

    inclusive_sum_->clear();
    inclusive_sum_->resize(pl_count);

    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& parts = particles.AccessParticles(pl_idx);

        auto const p_count = parts.GetCount();

        inclusive_sum_->operator[](pl_idx) = pl_idx == 0 ? p_count : p_count + inclusive_sum_->operator[](pl_idx - 1);
    }

    if (flags_read()) {
        auto const flags_data = flags_read.getData();
        flags_data->validateFlagCount(inclusive_sum_->back());

        flags_write.setData(flags_data, flags_read.version() + 1);
        flags_write();
    }

    return true;
}
