#include "PrepareSurfaceEvents2.h"

#include "mmcore/param/BoolParam.h"


megamol::thermodyn::PrepareSurfaceEvents2::PrepareSurfaceEvents2()
        : data_out_slot_("dataOut", "")
        , parts_in_slot_("partsIn", "")
        , table_in_slot_("tableIn", "")
        , show_all_parts_slot_("show all", "") {

    data_out_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &PrepareSurfaceEvents2::get_data_cb);
    data_out_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &PrepareSurfaceEvents2::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    parts_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&parts_in_slot_);

    table_in_slot_.SetCompatibleCall<stdplugin::datatools::table::TableDataCallDescription>();
    MakeSlotAvailable(&table_in_slot_);

    show_all_parts_slot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&show_all_parts_slot_);
}


megamol::thermodyn::PrepareSurfaceEvents2::~PrepareSurfaceEvents2() {
    this->Release();
}


bool megamol::thermodyn::PrepareSurfaceEvents2::create() {
    return true;
}


void megamol::thermodyn::PrepareSurfaceEvents2::release() {}


bool megamol::thermodyn::PrepareSurfaceEvents2::get_data_cb(core::Call& c) {
    auto data_out = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (data_out == nullptr)
        return false;

    auto parts_in = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto table_in = table_in_slot_.CallAs<stdplugin::datatools::table::TableDataCall>();
    if (table_in == nullptr)
        return false;

    auto const out_frame_id = data_out->FrameID();

    parts_in->SetFrameID(out_frame_id);
    if (!(*parts_in)(1))
        return false;
    if (!(*parts_in)(0))
        return false;

    table_in->SetFrameID(out_frame_id);
    if (!(*table_in)(1))
        return false;
    if (!(*table_in)(0))
        return false;

    if (parts_in->DataHash() != parts_in_data_hash_ || parts_in->FrameID() != frame_id_ ||
        table_in->DataHash() != table_in_data_hash_ /*|| table_in->GetFrameID() != frame_id_*/ || is_dirty()) {
        prepare_maps(*table_in);
        assert_data(*parts_in, *table_in, parts_in->FrameID());

        parts_in_data_hash_ = parts_in->DataHash();
        table_in_data_hash_ = table_in->DataHash();
        frame_id_ = parts_in->FrameID();
        ++parts_out_hash_;
        reset_dirty();
    }

    data_out->SetFrameCount(parts_in->FrameCount());
    data_out->SetFrameID(frame_id_);
    data_out->SetParticleListCount(part_data_.size());
    for (decltype(part_data_)::size_type pl_idx = 0; pl_idx < part_data_.size(); ++pl_idx) {
        auto& out_parts = data_out->AccessParticles(pl_idx);
        auto const& part_data = part_data_[pl_idx];

        auto const p_count = part_data.size() / 8;

        out_parts.SetCount(p_count);

        out_parts.SetIDData(core::moldyn::SimpleSphericalParticles::IDDATA_UINT32, part_data.data(), 8 * sizeof(float));
        out_parts.SetVertexData(
            core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, part_data.data() + 1, 8 * sizeof(float));
        out_parts.SetColourData(
            core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA, part_data.data() + 4, 8 * sizeof(float));
        out_parts.SetGlobalRadius(parts_in->AccessParticles(pl_idx).GetGlobalRadius());
    }

    return true;
}


bool megamol::thermodyn::PrepareSurfaceEvents2::get_extent_cb(core::Call& c) {
    auto data_out = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (data_out == nullptr)
        return false;

    auto parts_in = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto table_in = table_in_slot_.CallAs<stdplugin::datatools::table::TableDataCall>();
    if (table_in == nullptr)
        return false;

    auto const out_frame_id = data_out->FrameID();

    parts_in->SetFrameID(out_frame_id);
    if (!(*parts_in)(1))
        return false;

    table_in->SetFrameID(out_frame_id);
    if (!(*table_in)(1))
        return false;

    data_out->SetFrameCount(parts_in->FrameCount());
    data_out->SetFrameID(frame_id_);
    data_out->AccessBoundingBoxes() = parts_in->AccessBoundingBoxes();

    return true;
}


bool megamol::thermodyn::PrepareSurfaceEvents2::assert_data(
    core::moldyn::MultiParticleDataCall& in_parts, stdplugin::datatools::table::TableDataCall& in_table, int frame_id) {
    auto const show_all = show_all_parts_slot_.Param<core::param::BoolParam>()->Value();

    try {
        auto const event_list = frame_id_map_.at(frame_id);

        part_data_.clear();

        auto const pl_count = in_parts.GetParticleListCount();

        part_data_.resize(pl_count);

        for (std::remove_cv_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
            auto const& parts = in_parts.AccessParticles(pl_idx);
            auto& part_data = part_data_[pl_idx];

            auto const p_count = parts.GetCount();

            part_data.reserve(p_count);

            auto const idAcc = parts.GetParticleStore().GetIDAcc();
            auto const xAcc = parts.GetParticleStore().GetXAcc();
            auto const yAcc = parts.GetParticleStore().GetYAcc();
            auto const zAcc = parts.GetParticleStore().GetZAcc();
            auto const crAcc = parts.GetParticleStore().GetCRAcc();
            auto const cgAcc = parts.GetParticleStore().GetCGAcc();
            auto const cbAcc = parts.GetParticleStore().GetCBAcc();
            auto const caAcc = parts.GetParticleStore().GetCAAcc();

            for (std::remove_cv_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                auto const id = idAcc->Get_u32(p_idx);
                auto const fit = std::find(event_list.begin(), event_list.end(), id);
                if (show_all) {
                    part_data.push_back(*reinterpret_cast<float const*>(&id));
                    part_data.push_back(xAcc->Get_f(p_idx));
                    part_data.push_back(yAcc->Get_f(p_idx));
                    part_data.push_back(zAcc->Get_f(p_idx));
                    if (fit == event_list.end()) {
                        part_data.push_back(crAcc->Get_f(p_idx));
                        part_data.push_back(cgAcc->Get_f(p_idx));
                        part_data.push_back(cbAcc->Get_f(p_idx));
                        part_data.push_back(caAcc->Get_f(p_idx));
                    } else {
                        part_data.push_back(0.0f);
                        part_data.push_back(1.0f);
                        part_data.push_back(0.0f);
                        part_data.push_back(1.0f);
                    }
                } else {
                    if (fit != event_list.end()) {
                        auto const& [s_fid, e_fid, s_pos, e_pos] = event_map_.at(id);
                        part_data.push_back(*reinterpret_cast<float const*>(&id));
                        part_data.push_back(xAcc->Get_f(p_idx));
                        part_data.push_back(yAcc->Get_f(p_idx));
                        part_data.push_back(zAcc->Get_f(p_idx));
                        /*part_data.push_back(s_pos.x);
                        part_data.push_back(s_pos.y);
                        part_data.push_back(s_pos.z);*/
                        part_data.push_back(0.0f);
                        part_data.push_back(1.0f);
                        part_data.push_back(0.0f);
                        part_data.push_back(1.0f);
                    }
                }
            }
        }
    } catch (...) { return false; }

    return true;
}


void megamol::thermodyn::PrepareSurfaceEvents2::prepare_maps(
    stdplugin::datatools::table::TableDataCall const& in_table) {
    auto const col_count = in_table.GetColumnsCount();
    auto const row_count = in_table.GetRowsCount();
    auto const table_data = in_table.GetData();

    frame_id_map_.clear();
    // frame_id_map_.reserve(row_count);

    event_map_.clear();
    event_map_.reserve(row_count);

    for (std::remove_cv_t<decltype(row_count)> row = 0; row < row_count; ++row) {
        event_map_[in_table.GetData(0, row)] = {static_cast<int>(in_table.GetData(1, row)),
            static_cast<int>(in_table.GetData(2, row)),
            {in_table.GetData(3, row), in_table.GetData(4, row), in_table.GetData(5, row)},
            {in_table.GetData(6, row), in_table.GetData(7, row), in_table.GetData(8, row)}};

        for (auto i = in_table.GetData(1, row); i < in_table.GetData(2, row); ++i) {
            frame_id_map_[i].push_back(in_table.GetData(0, row));
        }
    }
}
