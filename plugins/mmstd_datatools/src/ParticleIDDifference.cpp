#include "ParticleIDDifference.h"

#include "mmcore/param/FloatParam.h"


megamol::stdplugin::datatools::ParticleIDDifference::ParticleIDDifference()
        : data_out_slot_("dataOut", "")
        , a_particles_slot_("dataAIn", "")
        , b_particles_slot_("dataBIn", "")
        , threshold_slot_("threshold", "") {
    data_out_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &ParticleIDDifference::get_data_cb);
    data_out_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &ParticleIDDifference::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    a_particles_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&a_particles_slot_);

    b_particles_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&b_particles_slot_);

    threshold_slot_ << new core::param::FloatParam(0.5f);
    MakeSlotAvailable(&threshold_slot_);
}


megamol::stdplugin::datatools::ParticleIDDifference::~ParticleIDDifference() {
    this->Release();
}


bool megamol::stdplugin::datatools::ParticleIDDifference::create() {
    return true;
}


void megamol::stdplugin::datatools::ParticleIDDifference::release() {}


bool megamol::stdplugin::datatools::ParticleIDDifference::get_data_cb(core::Call& c) {
    auto data_out = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (data_out == nullptr)
        return false;
    auto a_data_in = a_particles_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (a_data_in == nullptr)
        return false;
    auto b_data_in = b_particles_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (b_data_in == nullptr)
        return false;

    auto const req_frame_id = data_out->FrameID();
    a_data_in->SetFrameID(req_frame_id);
    b_data_in->SetFrameID(req_frame_id);
    if (!(*a_data_in)(0))
        return false;
    if (!(*b_data_in)(0))
        return false;

    if (a_data_in->DataHash() != a_in_data_hash_ || b_data_in->DataHash() != b_in_data_hash_ ||
        a_data_in->FrameID() != frame_id_ || b_data_in->FrameID() != frame_id_ || is_dirty()) {
        if (!assert_data(*a_data_in, *b_data_in))
            return false;

        a_in_data_hash_ = a_data_in->DataHash();
        b_in_data_hash_ = b_data_in->DataHash();
        frame_id_ = a_data_in->FrameID();
        reset_dirty();
        ++out_data_hash_;
    }

    auto const pl_count = data_.size();
    data_out->SetParticleListCount(pl_count);
    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto& parts = data_out->AccessParticles(pl_idx);
        auto const& data = data_[pl_idx];

        auto const p_count = data.size();
        parts.SetCount(p_count);

        parts.SetIDData(core::moldyn::SimpleSphericalParticles::IDDATA_UINT64, &data[0].id, sizeof(particle_t));
        parts.SetVertexData(core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, &data[0].x, sizeof(particle_t));
        parts.SetColourData(
            core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, &data[0].i_col, sizeof(particle_t));
        parts.SetDirData(core::moldyn::SimpleSphericalParticles::DIRDATA_FLOAT_XYZ, &data[0].dx, sizeof(particle_t));
    }

    data_out->SetFrameID(frame_id_);
    data_out->SetDataHash(out_data_hash_);

    return true;
}


bool megamol::stdplugin::datatools::ParticleIDDifference::get_extent_cb(core::Call& c) {
    auto data_out = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (data_out == nullptr)
        return false;
    auto a_data_in = a_particles_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (a_data_in == nullptr)
        return false;
    auto b_data_in = b_particles_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (b_data_in == nullptr)
        return false;

    auto const req_frame_id = data_out->FrameID();
    a_data_in->SetFrameID(req_frame_id);
    b_data_in->SetFrameID(req_frame_id);
    if (!(*a_data_in)(1))
        return false;
    if (!(*b_data_in)(1))
        return false;

    data_out->SetFrameCount(a_data_in->FrameCount());
    data_out->AccessBoundingBoxes() = a_data_in->AccessBoundingBoxes();

    return true;
}


bool megamol::stdplugin::datatools::ParticleIDDifference::assert_data(
    core::moldyn::MultiParticleDataCall& a_particles, core::moldyn::MultiParticleDataCall& b_particles) {
    auto const a_pl_count = a_particles.GetParticleListCount();
    auto const b_pl_count = b_particles.GetParticleListCount();

    if (a_pl_count != b_pl_count)
        return false;

    data_.resize(a_pl_count);

    auto const threshold = threshold_slot_.Param<core::param::FloatParam>()->Value();

    for (std::decay_t<decltype(a_pl_count)> pl_idx = 0; pl_idx < a_pl_count; ++pl_idx) {
        auto& data = data_[pl_idx];
        data.clear();
        // reserve some space

        auto const& a_parts = a_particles.AccessParticles(pl_idx);
        auto const& b_parts = b_particles.AccessParticles(pl_idx);

        auto const a_p_count = a_parts.GetCount();
        auto const b_p_count = b_parts.GetCount();

        auto const a_i_acc = a_parts.GetParticleStore().GetCRAcc();
        auto const b_i_acc = b_parts.GetParticleStore().GetCRAcc();

        auto const a_id_acc = a_parts.GetParticleStore().GetIDAcc();
        auto const a_x_acc = a_parts.GetParticleStore().GetXAcc();
        auto const a_y_acc = a_parts.GetParticleStore().GetYAcc();
        auto const a_z_acc = a_parts.GetParticleStore().GetZAcc();
        auto const a_dx_acc = a_parts.GetParticleStore().GetDXAcc();
        auto const a_dy_acc = a_parts.GetParticleStore().GetDYAcc();
        auto const a_dz_acc = a_parts.GetParticleStore().GetDZAcc();

        auto const b_id_acc = b_parts.GetParticleStore().GetIDAcc();
        auto const b_x_acc = b_parts.GetParticleStore().GetXAcc();
        auto const b_y_acc = b_parts.GetParticleStore().GetYAcc();
        auto const b_z_acc = b_parts.GetParticleStore().GetZAcc();
        auto const b_dx_acc = b_parts.GetParticleStore().GetDXAcc();
        auto const b_dy_acc = b_parts.GetParticleStore().GetDYAcc();
        auto const b_dz_acc = b_parts.GetParticleStore().GetDZAcc();

        std::unordered_map<uint64_t, uint64_t> a_map;
        a_map.reserve(a_p_count);
        std::vector<uint64_t> a_id_set;
        a_id_set.reserve(a_p_count);

        for (std::decay_t<decltype(a_p_count)> p_idx = 0; p_idx < a_p_count; ++p_idx) {
            a_map[a_id_acc->Get_u64(p_idx)] = p_idx;
            a_id_set.push_back(a_id_acc->Get_u64(p_idx));
        }
        std::sort(a_id_set.begin(), a_id_set.end());

        std::unordered_map<uint64_t, uint64_t> b_map;
        b_map.reserve(b_p_count);
        std::vector<uint64_t> b_id_set;
        b_id_set.reserve(b_p_count);

        for (std::decay_t<decltype(b_p_count)> p_idx = 0; p_idx < b_p_count; ++p_idx) {
            b_map[b_id_acc->Get_u64(p_idx)] = p_idx;
            b_id_set.push_back(b_id_acc->Get_u64(p_idx));
        }
        std::sort(b_id_set.begin(), b_id_set.end());


        std::list<uint64_t> first_id_list;
        std::set_difference(
            b_id_set.begin(), b_id_set.end(), a_id_set.begin(), a_id_set.end(), std::back_inserter(first_id_list));
        std::list<uint64_t> second_id_list;
        std::set_difference(
            a_id_set.begin(), a_id_set.end(), b_id_set.begin(), b_id_set.end(), std::back_inserter(second_id_list));

        std::list<uint64_t> id_list;
        std::merge(first_id_list.begin(), first_id_list.end(), second_id_list.begin(), second_id_list.end(),
            std::back_inserter(id_list));

        id_list.erase(std::unique(id_list.begin(), id_list.end()), id_list.end());

        for (auto const& el : id_list) {
            auto const fit = a_map.find(el);
            if (fit != a_map.end()) {
                auto const p_idx = fit->second;
                data.emplace_back(particle_t{a_id_acc->Get_u64(p_idx), a_x_acc->Get_f(p_idx), a_y_acc->Get_f(p_idx),
                    a_z_acc->Get_f(p_idx), 0.0f, a_dx_acc->Get_f(p_idx), a_dy_acc->Get_f(p_idx),
                    a_dz_acc->Get_f(p_idx)});
            } else {
                auto const fit = b_map.find(el);
                if (fit != b_map.end()) {
                    auto const p_idx = fit->second;
                    data.emplace_back(particle_t{b_id_acc->Get_u64(p_idx), b_x_acc->Get_f(p_idx), b_y_acc->Get_f(p_idx),
                        b_z_acc->Get_f(p_idx), 1.0f, b_dx_acc->Get_f(p_idx), b_dy_acc->Get_f(p_idx),
                        b_dz_acc->Get_f(p_idx)});
                }
            }
        }
    }

    return true;
}
