#include "IDBroker.h"

#include "mmcore/UniFlagCalls.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"


megamol::thermodyn::IDBroker::IDBroker()
        : out_flags_read_slot_("outFlagsRead", "")
        , out_flags_write_slot_("outFlagsWrite", "")
        , out_frame_id_slot_("outFrameID", "")
        , id_max_context_slot_("idMaxCtx", "")
        , id_sub_context_slot_("idSubCtx", "")
        , in_flags_read_slot_("inFlagsRead", "")
        , in_flags_write_slot_("inFlagsWrite", "") {
    out_flags_read_slot_.SetCallback(
        core::FlagCallRead_CPU::ClassName(), core::FlagCallRead_CPU::FunctionName(0), &IDBroker::flags_read_cb);
    out_flags_read_slot_.SetCallback(
        core::FlagCallRead_CPU::ClassName(), core::FlagCallRead_CPU::FunctionName(1), &IDBroker::flags_read_meta_cb);
    MakeSlotAvailable(&out_flags_read_slot_);

    out_flags_write_slot_.SetCallback(
        core::FlagCallWrite_CPU::ClassName(), core::FlagCallWrite_CPU::FunctionName(0), &IDBroker::flags_write_cb);
    out_flags_write_slot_.SetCallback(
        core::FlagCallWrite_CPU::ClassName(), core::FlagCallWrite_CPU::FunctionName(1), &IDBroker::flags_write_meta_cb);
    MakeSlotAvailable(&out_flags_write_slot_);

    out_frame_id_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &IDBroker::out_frame_id_data_cb);
    out_frame_id_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &IDBroker::out_frame_id_extent_cb);
    MakeSlotAvailable(&out_frame_id_slot_);

    id_max_context_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&id_max_context_slot_);

    id_sub_context_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&id_sub_context_slot_);

    in_flags_read_slot_.SetCompatibleCall<core::FlagCallRead_CPUDescription>();
    MakeSlotAvailable(&in_flags_read_slot_);

    in_flags_write_slot_.SetCompatibleCall<core::FlagCallWrite_CPUDescription>();
    MakeSlotAvailable(&in_flags_write_slot_);
}


megamol::thermodyn::IDBroker::~IDBroker() {
    this->Release();
}


bool megamol::thermodyn::IDBroker::create() {
    const int num = 10;
    flag_col_ = std::make_shared<core::FlagCollection_CPU>();
    flag_col_->flags = std::make_shared<core::FlagStorage::FlagVectorType>(num, core::FlagStorage::ENABLED);
    return true;
}


void megamol::thermodyn::IDBroker::release() {}


bool megamol::thermodyn::IDBroker::flags_read_cb(core::Call& c) {
    auto fcr = dynamic_cast<core::FlagCallRead_CPU*>(&c);
    if (fcr == nullptr)
        return false;

    auto id_max_data = id_max_context_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (id_max_data == nullptr)
        return false;

    auto id_sub_data = id_sub_context_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (id_sub_data == nullptr)
        return false;

    auto in_fcr = in_flags_read_slot_.CallAs<core::FlagCallRead_CPU>();
    if (in_fcr == nullptr)
        return false;

    if (!(*in_fcr)(core::FlagCallRead_CPU::CallGetData))
        return false;

    id_max_data->SetFrameID(out_frame_id_);
    if (!(*id_max_data)(0))
        return false;

    id_sub_data->SetFrameID(out_frame_id_);
    if (!(*id_sub_data)(0))
        return false;

    if (id_max_data->DataHash() != in_data_hash_ || id_max_data->FrameID() != in_frame_id_) {
        auto const pl_count = id_max_data->GetParticleListCount();

        id_maps_.clear();
        id_maps_.resize(pl_count);

        uint64_t totalCount = 0;
        for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
            auto const& parts = id_max_data->AccessParticles(pl_idx);
            auto const p_count = parts.GetCount();
            totalCount += p_count;

            auto const id_acc = parts.GetParticleStore().GetIDAcc();

            auto& id_map = id_maps_[pl_idx];
            id_map.reserve(p_count);

            for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                id_map[id_acc->Get_u64(p_idx)] = p_idx;
            }
        }

        in_fcr->getData()->validateFlagCount(totalCount);

        in_data_hash_ = id_max_data->DataHash();
        in_frame_id_ = id_max_data->FrameID();
    }

    if (id_sub_data->DataHash() != in_sub_data_hash_ || id_sub_data->FrameID() != in_sub_frame_id_) {
        auto const pl_count = id_sub_data->GetParticleListCount();

        idx_maps_.clear();
        idx_maps_.resize(pl_count);

        prefix_count_.resize(pl_count);

        for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
            auto const& parts = id_sub_data->AccessParticles(pl_idx);
            auto const p_count = parts.GetCount();

            prefix_count_[pl_idx] = pl_idx == 0 ? p_count : prefix_count_[pl_idx - 1] + p_count;

            auto const id_acc = parts.GetParticleStore().GetIDAcc();

            auto& idx_map = idx_maps_[pl_idx];
            idx_map.reserve(p_count);

            for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                idx_map[p_idx] = id_acc->Get_u64(p_idx);
            }
        }

        in_sub_data_hash_ = id_sub_data->DataHash();
        in_sub_frame_id_ = id_sub_data->FrameID();
    }

    // validate size
    // establish translation map

    fcr->setData(flag_col_, version_);

    return true;
}


bool megamol::thermodyn::IDBroker::flags_read_meta_cb(core::Call& c) {
    return true;
}


bool megamol::thermodyn::IDBroker::flags_write_cb(core::Call& c) {
    auto fcw = dynamic_cast<core::FlagCallWrite_CPU*>(&c);
    if (fcw == nullptr)
        return false;

    auto in_fcr = in_flags_read_slot_.CallAs<core::FlagCallRead_CPU>();
    if (in_fcr == nullptr)
        return false;

    auto in_fcw = in_flags_write_slot_.CallAs<core::FlagCallRead_CPU>();
    if (in_fcw == nullptr)
        return false;

    if (!(*in_fcr)(0))
        return false;

    bool flags_changed = fcw->version() > version_;
    if (flags_changed) {
        flag_col_ = fcw->getData();
        version_ = fcw->version();

        auto const in_flag_col = in_fcr->getData();
        auto& in_flags = *in_flag_col->flags;

        auto const& flags = *flag_col_->flags;
        for (std::decay_t<decltype(flags)>::size_type f_idx = 0; f_idx < flags.size(); ++f_idx) {
            try {
                auto const fit = std::find_if(
                    prefix_count_.begin(), prefix_count_.end(), [&f_idx](auto const el) { return f_idx < el; });
                if (fit != prefix_count_.end()) {
                    auto const pl_idx = std::distance(prefix_count_.begin(), fit);
                    auto const id = idx_maps_[pl_idx].at(f_idx);
                    auto const in_f_idx = id_maps_[pl_idx].at(id);
                    // set incoming flags storage
                    in_flags[in_f_idx] = flags[f_idx];
                }
            } catch (...) {}
        }

        in_fcw->setData(in_flag_col, in_fcr->version() + 1);
        (*in_fcw)(0);
    }

    return true;
}


bool megamol::thermodyn::IDBroker::flags_write_meta_cb(core::Call& c) {
    return true;
}


bool megamol::thermodyn::IDBroker::out_frame_id_data_cb(core::Call& c) {
    auto out_frame_id_call = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_frame_id_call == nullptr)
        return false;

    auto id_sub_data = id_sub_context_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (id_sub_data == nullptr)
        return false;

    out_frame_id_ = out_frame_id_call->FrameID();
    id_sub_data->SetFrameID(out_frame_id_);
    if (!(*id_sub_data)(1))
        return false;
    if (!(*id_sub_data)(0))
        return false;

    *out_frame_id_call = *id_sub_data;
    id_sub_data->SetUnlocker(nullptr);

    return true;
}


bool megamol::thermodyn::IDBroker::out_frame_id_extent_cb(core::Call& c) {
    auto out_frame_id_call = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (out_frame_id_call == nullptr)
        return false;

    auto id_sub_data = id_sub_context_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (id_sub_data == nullptr)
        return false;

    out_frame_id_ = out_frame_id_call->FrameID();
    id_sub_data->SetFrameID(out_frame_id_);
    if (!(*id_sub_data)(1))
        return false;
    if (!(*id_sub_data)(0))
        return false;

    *out_frame_id_call = *id_sub_data;
    id_sub_data->SetUnlocker(nullptr);

    return true;
}
