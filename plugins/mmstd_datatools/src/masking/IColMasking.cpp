#include "IColMasking.h"

#include "mmcore/param/FloatParam.h"


megamol::stdplugin::datatools::masking::IColMasking::IColMasking()
        : data_out_slot_("dataOut", "")
        , data_in_slot_("dataIn", "")
        , flags_read_slot_("flagsRead", "")
        , flags_write_slot_("flagsWrite", "")
        , min_val_slot_("minVal", "")
        , max_val_slot_("maxVal", "") {
    data_out_slot_.SetCallback<core::moldyn::MultiParticleDataCall, 0>(&IColMasking::get_data_cb);
    data_out_slot_.SetCallback<core::moldyn::MultiParticleDataCall, 1>(&IColMasking::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    flags_read_slot_.SetCompatibleCall<core::FlagCallRead_CPUDescription>();
    MakeSlotAvailable(&flags_read_slot_);

    flags_write_slot_.SetCompatibleCall<core::FlagCallWrite_CPUDescription>();
    MakeSlotAvailable(&flags_write_slot_);

    min_val_slot_ << new core::param::FloatParam(0.0f);
    MakeSlotAvailable(&min_val_slot_);

    max_val_slot_ << new core::param::FloatParam(1.0f);
    MakeSlotAvailable(&max_val_slot_);
}


megamol::stdplugin::datatools::masking::IColMasking::~IColMasking() {
    this->Release();
}


bool megamol::stdplugin::datatools::masking::IColMasking::create() {
    return true;
}


void megamol::stdplugin::datatools::masking::IColMasking::release() {}


bool megamol::stdplugin::datatools::masking::IColMasking::get_data_cb(core::Call& c) {
    auto data_out = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (data_out == nullptr)
        return false;

    auto data_in = data_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (data_out == nullptr)
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

    if (flags_read->hasUpdate() || data_in->DataHash() != in_data_hash_ || data_in->FrameID() != frame_id_ ||
        is_dirty()) {
        if (!assert_data(*data_in, *flags_read, *flags_write))
            return false;
        in_data_hash_ = data_in->DataHash();
        frame_id_ = data_in->FrameID();
        reset_dirty();
    }

    *data_out = *data_in;
    data_in->SetUnlocker(nullptr, false);

    return true;
}


bool megamol::stdplugin::datatools::masking::IColMasking::get_extent_cb(core::Call& c) {
    auto data_out = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (data_out == nullptr)
        return false;

    auto data_in = data_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (data_out == nullptr)
        return false;

    data_in->SetFrameID(data_out->FrameID());
    if (!(*data_in)(1))
        return false;

    data_out->SetFrameCount(data_in->FrameCount());
    data_out->AccessBoundingBoxes() = data_in->AccessBoundingBoxes();

    return true;
}


bool megamol::stdplugin::datatools::masking::IColMasking::assert_data(core::moldyn::MultiParticleDataCall& particles,
    core::FlagCallRead_CPU& flags_read, core::FlagCallWrite_CPU& flags_write) {
    auto const min_val = min_val_slot_.Param<core::param::FloatParam>()->Value();
    auto const max_val = max_val_slot_.Param<core::param::FloatParam>()->Value();

    if (flags_read(0)) {
        auto const flags_ptr = flags_read.getData();
        auto& flags = *(flags_ptr->flags);

        auto const pl_count = particles.GetParticleListCount();

        uint64_t flag_idx = 0;
        for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
            auto const& parts = particles.AccessParticles(pl_idx);

            auto const p_count = parts.GetCount();

            auto const i_acc = parts.GetParticleStore().GetCRAcc();

            for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx, ++flag_idx) {
                auto const val = i_acc->Get_f(p_idx);

                if (min_val > val || val > max_val) {
                    flags[flag_idx] = core::FlagStorage::FILTERED;
                }
            }
        }

        flags_write.setData(flags_ptr, flags_read.version() + 1);
        flags_write(0);
    }

    return true;
}
