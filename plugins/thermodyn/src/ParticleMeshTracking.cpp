#include "ParticleMeshTracking.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"


megamol::thermodyn::ParticleMeshTracking::ParticleMeshTracking()
        : data_out_slot_("dataOut", "")
        , parts_in_slot_("partsIn", "")
        , all_frames_slot_("frames::all", "")
        , min_frames_slot_("frames::min", "")
        , max_frames_slot_("frames::max", "") {
    data_out_slot_.SetCallback<stdplugin::datatools::table::TableDataCall, 0>(&ParticleMeshTracking::get_data_cb);
    data_out_slot_.SetCallback<stdplugin::datatools::table::TableDataCall, 1>(&ParticleMeshTracking::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    parts_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&parts_in_slot_);

    all_frames_slot_ << new core::param::BoolParam(true);
    MakeSlotAvailable(&all_frames_slot_);

    min_frames_slot_ << new core::param::IntParam(0, 0);
    MakeSlotAvailable(&min_frames_slot_);

    max_frames_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&max_frames_slot_);
}


megamol::thermodyn::ParticleMeshTracking::~ParticleMeshTracking() {
    this->Release();
}


bool megamol::thermodyn::ParticleMeshTracking::create() {
    return true;
}


void megamol::thermodyn::ParticleMeshTracking::release() {}


bool megamol::thermodyn::ParticleMeshTracking::get_data_cb(core::Call& c) {
    auto data_out = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (data_out == nullptr)
        return false;

    auto parts_in = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    if (!(*parts_in)(0))
        return false;

    if (parts_in->DataHash() != in_data_hash_) {
        if (!assert_data(*parts_in))
            return false;
        in_data_hash_ = parts_in->DataHash();
        ++out_data_hash_;
    }

    if (row_count_ == 0) {
        data_out->Set(0, 0, nullptr, nullptr);
        return true;
    }

    data_out->Set(col_count_, row_count_, infos_.data(), data_.data());
    data_out->SetDataHash(out_data_hash_);
    data_out->SetFrameID(0);

    return true;
}


bool megamol::thermodyn::ParticleMeshTracking::get_extent_cb(core::Call& c) {
    auto data_out = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (data_out == nullptr)
        return false;

    auto parts_in = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    if (!(*parts_in)(1))
        return false;

    data_out->SetFrameCount(1);

    return true;
}


std::tuple<unsigned int, unsigned int> min_max_frame(
    unsigned int min_val, unsigned int max_val, unsigned int frame_count) {
    if (frame_count < 2) {
        return std::make_tuple(0, 0);
    }
    if (frame_count < 3) {
        return std::make_tuple(0, 1);
    }
    if (min_val > max_val) {
        std::swap(min_val, max_val);
    }

    if (min_val < 0)
        min_val = 0;

    if (min_val >= frame_count)
        min_val = frame_count - 2;

    if (max_val < 0)
        max_val = 0;

    if (max_val >= frame_count)
        max_val = frame_count - 1;

    return std::make_tuple(min_val, max_val);
}


bool megamol::thermodyn::ParticleMeshTracking::assert_data(core::moldyn::MultiParticleDataCall& particles) {
    auto const pl_count = particles.GetParticleListCount();
    if (pl_count != 1)
        return false;

    auto const min_frame = min_frames_slot_.Param<core::param::IntParam>()->Value();
    auto const max_frame = max_frames_slot_.Param<core::param::IntParam>()->Value();
    auto const all_frames = all_frames_slot_.Param<core::param::BoolParam>()->Value();

    auto min_f = 0;
    auto max_f = particles.FrameCount() - 1;

    if (!all_frames) {
        std::tie(min_f, max_f) = min_max_frame(min_frame, max_frame, particles.FrameCount());
    }

    if (min_f == 0 && max_f == 0)
        return false;

    std::unordered_map<uint64_t /*id*/, std::tuple<uint8_t /*state*/, uint32_t /*frame idx*/>> p_state;

    std::list<std::tuple<uint64_t, uint32_t, uint32_t>> events;

    for (unsigned int f_idx = min_f; f_idx <= max_f; ++f_idx) {
        bool got_frame = false;
        do {
            particles.SetFrameID(f_idx);
            got_frame = particles(0);
        } while (!got_frame || particles.FrameID() != f_idx);

        auto const& parts = particles.AccessParticles(0);

        auto const p_count = parts.GetCount();

        auto const id_acc = parts.GetParticleStore().GetIDAcc();
        auto const i_acc = parts.GetParticleStore().GetCRAcc();

        p_state.reserve(p_count);

        for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
            auto const id = id_acc->Get_f(p_idx);
            auto const state = static_cast<uint8_t>(i_acc->Get_f(p_idx));

            auto const fit = p_state.find(id);
            if (fit == p_state.end()) {
                if (state == 0) {
                    p_state[id] = std::make_tuple(0, f_idx);
                }
                continue;
            }
            {
                if (std::get<0>(p_state[id]) == 0 && state == 1) {
                    p_state[id] = std::make_tuple(1, f_idx);
                    continue;
                }
                if (std::get<0>(p_state[id]) == 1 && state == 0) {
                    events.push_back(std::make_tuple(id, std::get<1>(p_state[id]), f_idx));
                    p_state[id] = std::make_tuple(0, f_idx);
                    continue;
                }
            }
        }
    }

    row_count_ = events.size();
    col_count_ = 3;
    if (row_count_ == 0)
        return true;

    infos_.clear();
    data_.clear();
    data_.resize(row_count_ * col_count_);

    auto min_max_id = std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());
    auto min_max_start_f = std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());
    auto min_max_end_f = std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());

    uint64_t counter = 0;
    for (auto const& [id, start_f, end_f] : events) {
        data_[0 + counter * col_count_] = id;
        if (data_[0 + counter * col_count_] < min_max_id.first) {
            min_max_id.first = data_[0 + counter * col_count_];
        }
        if (data_[0 + counter * col_count_] > min_max_id.second) {
            min_max_id.second = data_[0 + counter * col_count_];
        }
        data_[1 + counter * col_count_] = start_f;
        if (data_[1 + counter * col_count_] < min_max_start_f.first) {
            min_max_start_f.first = data_[1 + counter * col_count_];
        }
        if (data_[1 + counter * col_count_] > min_max_start_f.second) {
            min_max_start_f.second = data_[1 + counter * col_count_];
        }
        data_[2 + counter * col_count_] = end_f;
        if (data_[2 + counter * col_count_] < min_max_end_f.first) {
            min_max_end_f.first = data_[2 + counter * col_count_];
        }
        if (data_[2 + counter * col_count_] > min_max_end_f.second) {
            min_max_end_f.second = data_[2 + counter * col_count_];
        }

        ++counter;
    }

    decltype(infos_)::value_type info;
    info.SetName("id");
    info.SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
    info.SetMinimumValue(min_max_id.first);
    info.SetMaximumValue(min_max_id.second);
    infos_.push_back(info);
    info.SetName("start_frame_id");
    info.SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
    info.SetMinimumValue(min_max_start_f.first);
    info.SetMaximumValue(min_max_start_f.second);
    infos_.push_back(info);
    info.SetName("end_frame_id");
    info.SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
    info.SetMinimumValue(min_max_end_f.first);
    info.SetMaximumValue(min_max_end_f.second);
    infos_.push_back(info);

    return true;
}
