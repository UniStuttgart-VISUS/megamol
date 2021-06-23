#include "PathSelection.h"


megamol::thermodyn::PathSelection::PathSelection()
        : data_out_slot_("dataOut", "")
        , data_in_slot_("dataIn", "")
        , parts_in_slot_("partsIn", "")
        , flags_read_slot_("flagsRead", "") {
    data_out_slot_.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &PathSelection::get_data_cb);
    data_out_slot_.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &PathSelection::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&data_in_slot_);

    parts_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&parts_in_slot_);

    flags_read_slot_.SetCompatibleCall<core::FlagCallRead_CPUDescription>();
    MakeSlotAvailable(&flags_read_slot_);
}


megamol::thermodyn::PathSelection::~PathSelection() {
    this->Release();
}


bool megamol::thermodyn::PathSelection::create() {
    return true;
}


void megamol::thermodyn::PathSelection::release() {}


bool megamol::thermodyn::PathSelection::get_data_cb(core::Call& c) {
    auto data_out = dynamic_cast<mesh::CallMesh*>(&c);
    if (data_out == nullptr)
        return false;

    auto data_in = data_in_slot_.CallAs<mesh::CallMesh>();
    if (data_in == nullptr)
        return false;

    auto parts_in = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto flags_read = flags_read_slot_.CallAs<core::FlagCallRead_CPU>();
    if (flags_read == nullptr)
        return false;

    auto out_meta = data_out->getMetaData();
    auto in_meta = data_in->getMetaData();
    in_meta.m_frame_ID = out_meta.m_frame_ID;
    if (!(*data_in)(0))
        return false;

    parts_in->SetFrameID(out_meta.m_frame_ID);
    if (!(*parts_in)(0))
        return false;

    if (!(*flags_read)(0))
        return false;

    if (data_in->hasUpdate() || flags_read->hasUpdate()) {
        if (!assert_data(*data_in, *parts_in, *flags_read))
            return false;
    }

    data_out->setData(mesh_col_, out_data_hash_);

    return true;
}


bool megamol::thermodyn::PathSelection::get_extent_cb(core::Call& c) {
    auto data_out = dynamic_cast<mesh::CallMesh*>(&c);
    if (data_out == nullptr)
        return false;

    auto data_in = data_in_slot_.CallAs<mesh::CallMesh>();
    if (data_in == nullptr)
        return false;

    auto parts_in = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto flags_read = flags_read_slot_.CallAs<core::FlagCallRead_CPU>();
    if (flags_read == nullptr)
        return false;

    auto out_meta = data_out->getMetaData();
    auto in_meta = data_in->getMetaData();
    in_meta.m_frame_ID = out_meta.m_frame_ID;
    if (!(*data_in)(1))
        return false;

    parts_in->SetFrameID(out_meta.m_frame_ID);
    if (!(*parts_in)(1))
        return false;

    if (!(*flags_read)(1))
        return false;

    in_meta = data_in->getMetaData();
    in_meta.m_frame_ID = out_meta.m_frame_ID;
    data_out->setMetaData(in_meta);

    return true;
}


bool megamol::thermodyn::PathSelection::assert_data(
    mesh::CallMesh& mesh, core::moldyn::MultiParticleDataCall& particles, core::FlagCallRead_CPU& flags) {
    mesh_col_ = std::make_shared<mesh::MeshDataAccessCollection>();

    auto const data = mesh.getData();

    auto const meshes = data->accessMeshes();

    auto const flags_data = flags.getData();

    auto const parts = particles.AccessParticles(0); //< TOXIC

    auto const id_acc = parts.GetParticleStore().GetIDAcc();

    bool found_picks = false;

    for (std::decay_t<decltype(*flags_data->flags)>::size_type f_idx = 0; f_idx < flags_data->flags->size(); ++f_idx) {
        if (flags_data->flags->operator[](f_idx) == core::FlagStorage::SELECTED) {
            auto const idx = id_acc->Get_u64(f_idx);
            auto const fit = meshes.find(std::to_string(idx));
            if (fit != meshes.end()) {
                mesh_col_->addMesh(fit->first, fit->second);
                found_picks = true;
            }
        }
    }

    if (found_picks)
        ++out_data_hash_;

    return true;
}
