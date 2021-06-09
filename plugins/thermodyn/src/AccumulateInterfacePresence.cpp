#include "AccumulateInterfacePresence.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"


megamol::thermodyn::AccumulateInterfacePresence::AccumulateInterfacePresence()
        : data_in_slot_("dataIn", "")
        , data_out_slot_("dataOut", "")
        , frame_count_slot_("frame count", "")
        , toggle_all_frames_slot_("toggle all frames", "") {
    data_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    data_out_slot_.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(0), &AccumulateInterfacePresence::get_data_cb);
    data_out_slot_.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(1), &AccumulateInterfacePresence::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    frame_count_slot_ << new core::param::IntParam(10, 1);
    MakeSlotAvailable(&frame_count_slot_);

    toggle_all_frames_slot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&toggle_all_frames_slot_);
}


megamol::thermodyn::AccumulateInterfacePresence::~AccumulateInterfacePresence() {
    this->Release();
}


bool megamol::thermodyn::AccumulateInterfacePresence::create() {
    return true;
}


void megamol::thermodyn::AccumulateInterfacePresence::release() {}


bool megamol::thermodyn::AccumulateInterfacePresence::get_data_cb(core::Call& c) {
    auto out_data = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (out_data == nullptr)
        return false;
    auto in_data = data_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    // set frame id
    in_data->SetFrameID(out_data->GetFrameID());
    if (!(*in_data)(1))
        return false;

    if (!(*in_data)(0))
        return false;

    if (is_dirty() || in_data_hash_ != in_data->DataHash() || frame_id_ != in_data->FrameID()) {
        assert_data(*in_data);

        frame_id_ = in_data->FrameID();
        in_data_hash_ = in_data->DataHash();
        ++out_data_hash_;
        reset_dirty();
    }

    // set outdata
    out_data->Set(infos_.size(), data_.size() / infos_.size(), infos_.data(), data_.data());
    out_data->SetDataHash(out_data_hash_);
    out_data->SetFrameCount(1);
    out_data->SetFrameID(1);

    return true;
}


bool megamol::thermodyn::AccumulateInterfacePresence::get_extent_cb(core::Call& c) {
    auto out_data = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (out_data == nullptr)
        return false;
    auto in_data = data_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    // set frame id
    in_data->SetFrameID(out_data->GetFrameID());
    if (!(*in_data)(1))
        return false;

    // set outdata
    out_data->SetDataHash(out_data_hash_);
    out_data->SetFrameCount(1);
    out_data->SetFrameID(1);

    return true;
}


bool megamol::thermodyn::AccumulateInterfacePresence::assert_data(core::moldyn::MultiParticleDataCall& points) {
    auto const frame_count = frame_count_slot_.Param<core::param::IntParam>()->Value();
    auto const all_frames = toggle_all_frames_slot_.Param<core::param::BoolParam>()->Value();

    auto const start_frame_id = all_frames ? 0 : points.FrameID();
    auto const end_frame_id = all_frames ? frame_count : start_frame_id + frame_count;

    auto const total_frame_count = points.FrameCount();

    // auto current_frame_id = start_frame_id;

    for (auto fid = start_frame_id; fid < end_frame_id && fid < total_frame_count; ++fid) {
        auto current_frame_id = -1;
        do {
            points.SetFrameID(fid, true);
            points(1);
            points(0);
            current_frame_id = points.FrameID();
        } while (current_frame_id != fid);

        auto const pl_count = points.GetParticleListCount();

        for (std::remove_cv_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
            auto const& particles = points.AccessParticles(pl_idx);

            auto const p_count = particles.GetCount();

            auto idAcc = particles.GetParticleStore().GetIDAcc();
            auto crAcc = particles.GetParticleStore().GetCRAcc();
            auto xAcc = particles.GetParticleStore().GetXAcc();
            auto yAcc = particles.GetParticleStore().GetYAcc();
            auto zAcc = particles.GetParticleStore().GetZAcc();

            for (std::remove_cv_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                auto const id = idAcc->Get_u64(p_idx);
                auto const val = crAcc->Get_f(p_idx);
                auto const x_pos = xAcc->Get_f(p_idx);
                auto const y_pos = yAcc->Get_f(p_idx);
                auto const z_pos = zAcc->Get_f(p_idx);

                auto& [state, s_fid, e_fid, s_pos, e_pos] = state_cache_[id];

                if (state == 0) {
                    s_fid = -1;
                    e_fid = -1;
                    s_pos = {-1, -1, -1};
                    e_pos = {-1, -1, -1};
                }

                if (val == 3) {
                    if (state != 3) {
                        state = 3;
                        s_fid = fid;
                        s_pos = {x_pos, y_pos, z_pos};
                    }
                } else if (val == 1) {
                    if (state == 4) {
                        state = 1;
                        e_fid = fid;
                        e_pos = {x_pos, y_pos, z_pos};
                        states_.push_back({id, s_fid, e_fid, s_pos, e_pos});
                    }
                    if (state == 3) {
                        state = 4;
                    }
                }

                //if (state == 0) {
                //    s_fid = -1;
                //    e_fid = -1;
                //    s_pos = {-1, -1, -1};
                //    e_pos = {-1, -1, -1};
                //    // not yet created
                //    if (val == 0.f) {
                //        state = 1;
                //    } else {
                //        state = 2;
                //        s_fid = fid;
                //        s_pos = {x_pos, y_pos, z_pos};
                //    }
                //} else if (state == 1) {
                //    if (val == 1.f) {
                //        state = 2;
                //        s_fid = fid;
                //        s_pos = {x_pos, y_pos, z_pos};
                //    }
                //} else if (state == 2) {
                //    if (val == 0.f) {
                //        state = 1;
                //        e_fid = fid;
                //        e_pos = {x_pos, y_pos, z_pos};
                //        states_.push_back({id, s_fid, e_fid, s_pos, e_pos});
                //        s_fid = -1;
                //        e_fid = -1;
                //    }
                //}
            }
        }
    }

    /*states_.clear();
    states_.reserve(state_cache_.size());
    for (auto const& [key, value] : state_cache_) {
        auto const& [state, s_fid, e_fid] = value;
        if (state == 3) {
            states_.push_back({key, s_fid, e_fid});
        }
    }*/

    auto const minmax_id = std::minmax_element(states_.begin(), states_.end(),
        [](auto const& lhs, auto const& rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });
    auto const minmax_sfid = std::minmax_element(states_.begin(), states_.end(),
        [](auto const& lhs, auto const& rhs) { return std::get<1>(lhs) < std::get<1>(rhs); });
    auto const minmax_efid = std::minmax_element(states_.begin(), states_.end(),
        [](auto const& lhs, auto const& rhs) { return std::get<2>(lhs) < std::get<2>(rhs); });
    auto const minmax_sx = std::minmax_element(states_.begin(), states_.end(),
        [](auto const& lhs, auto const& rhs) { return std::get<3>(lhs)[0] < std::get<3>(rhs)[0]; });
    auto const minmax_sy = std::minmax_element(states_.begin(), states_.end(),
        [](auto const& lhs, auto const& rhs) { return std::get<3>(lhs)[1] < std::get<3>(rhs)[1]; });
    auto const minmax_sz = std::minmax_element(states_.begin(), states_.end(),
        [](auto const& lhs, auto const& rhs) { return std::get<3>(lhs)[2] < std::get<3>(rhs)[2]; });
    auto const minmax_ex = std::minmax_element(states_.begin(), states_.end(),
        [](auto const& lhs, auto const& rhs) { return std::get<4>(lhs)[0] < std::get<4>(rhs)[0]; });
    auto const minmax_ey = std::minmax_element(states_.begin(), states_.end(),
        [](auto const& lhs, auto const& rhs) { return std::get<4>(lhs)[1] < std::get<4>(rhs)[1]; });
    auto const minmax_ez = std::minmax_element(states_.begin(), states_.end(),
        [](auto const& lhs, auto const& rhs) { return std::get<4>(lhs)[2] < std::get<4>(rhs)[2]; });


    auto const column_count = 9;
    auto const row_count = states_.size();

    data_.clear();
    data_.reserve(column_count * row_count);

    for (auto const& [id, s_fid, e_fid, s_pos, e_pos] : states_) {
        data_.push_back(id);
        data_.push_back(s_fid);
        data_.push_back(e_fid);
        data_.push_back(s_pos[0]);
        data_.push_back(s_pos[1]);
        data_.push_back(s_pos[2]);
        data_.push_back(e_pos[0]);
        data_.push_back(e_pos[1]);
        data_.push_back(e_pos[2]);
    }

    infos_.clear();
    infos_.resize(column_count);
    infos_[0].SetName("id");
    infos_[0].SetMinimumValue(std::get<0>(*minmax_id.first));
    infos_[0].SetMaximumValue(std::get<0>(*minmax_id.second));
    infos_[0].SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
    infos_[1].SetName("start_frame_id");
    infos_[1].SetMinimumValue(std::get<1>(*minmax_sfid.first));
    infos_[1].SetMaximumValue(std::get<1>(*minmax_sfid.second));
    infos_[1].SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
    infos_[2].SetName("end_frame_id");
    infos_[2].SetMinimumValue(std::get<2>(*minmax_efid.first));
    infos_[2].SetMaximumValue(std::get<2>(*minmax_efid.second));
    infos_[2].SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
    infos_[3].SetName("start_x_pos");
    infos_[3].SetMinimumValue(std::get<3>(*minmax_sx.first)[0]);
    infos_[3].SetMaximumValue(std::get<3>(*minmax_sx.second)[0]);
    infos_[3].SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
    infos_[4].SetName("start_y_pos");
    infos_[4].SetMinimumValue(std::get<3>(*minmax_sy.first)[1]);
    infos_[4].SetMaximumValue(std::get<3>(*minmax_sy.second)[1]);
    infos_[4].SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
    infos_[5].SetName("start_z_pos");
    infos_[5].SetMinimumValue(std::get<3>(*minmax_sz.first)[2]);
    infos_[5].SetMaximumValue(std::get<3>(*minmax_sz.second)[2]);
    infos_[5].SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
    infos_[6].SetName("end_x_pos");
    infos_[6].SetMinimumValue(std::get<4>(*minmax_ex.first)[0]);
    infos_[6].SetMaximumValue(std::get<4>(*minmax_ex.second)[0]);
    infos_[6].SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
    infos_[7].SetName("end_y_pos");
    infos_[7].SetMinimumValue(std::get<4>(*minmax_ey.first)[1]);
    infos_[7].SetMaximumValue(std::get<4>(*minmax_ey.second)[1]);
    infos_[7].SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
    infos_[8].SetName("end_z_pos");
    infos_[8].SetMinimumValue(std::get<4>(*minmax_ez.first)[2]);
    infos_[8].SetMaximumValue(std::get<4>(*minmax_ez.second)[2]);
    infos_[8].SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);

    return true;
}
