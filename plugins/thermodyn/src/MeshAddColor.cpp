#include "MeshAddColor.h"

#include "mmstd_datatools/TFUtils.h"


megamol::thermodyn::MeshAddColor::MeshAddColor()
        : data_out_slot_("dataOut", "")
        , mesh_in_slot_("meshIn", "")
        , parts_in_slot_("partsIn", "")
        , tf_in_slot_("tfIn", "") {
    data_out_slot_.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &MeshAddColor::get_data_cb);
    data_out_slot_.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &MeshAddColor::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    mesh_in_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&mesh_in_slot_);

    parts_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&parts_in_slot_);

    tf_in_slot_.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    MakeSlotAvailable(&tf_in_slot_);
}


megamol::thermodyn::MeshAddColor::~MeshAddColor() {
    this->Release();
}


bool megamol::thermodyn::MeshAddColor::create() {
    return true;
}


void megamol::thermodyn::MeshAddColor::release() {}


bool megamol::thermodyn::MeshAddColor::get_data_cb(core::Call& c) {
    auto data_out = dynamic_cast<mesh::CallMesh*>(&c);
    if (data_out == nullptr)
        return false;

    auto mesh_in = mesh_in_slot_.CallAs<mesh::CallMesh>();
    if (mesh_in == nullptr)
        return false;

    auto parts_in = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto tf_in = tf_in_slot_.CallAs<core::view::CallGetTransferFunction>();

    auto meta = data_out->getMetaData();
    auto mesh_meta = mesh_in->getMetaData();
    mesh_meta.m_frame_ID = meta.m_frame_ID;
    mesh_in->setMetaData(mesh_meta);
    if (!(*mesh_in)(0))
        return false;

    parts_in->SetFrameID(meta.m_frame_ID);
    if (!(*parts_in)(0))
        return false;

    if (tf_in) {
        (*tf_in)(0);
    }

    if (mesh_in->hasUpdate() || parts_in->DataHash() != in_data_hash_ || parts_in->FrameID() != frame_id_ ||
        (tf_in != nullptr && tf_in->IsDirty())) {
        if (!assert_data(*mesh_in, *parts_in, tf_in))
            return false;
        in_data_hash_ = parts_in->DataHash();
        frame_id_ = parts_in->FrameID();
        if (tf_in) {
            tf_in->ResetDirty();
        }
        ++version;
    }

    meta.m_frame_ID = frame_id_;
    data_out->setMetaData(meta);
    data_out->setData(mesh_col_, version);

    return true;
}


bool megamol::thermodyn::MeshAddColor::get_extent_cb(core::Call& c) {
    auto data_out = dynamic_cast<mesh::CallMesh*>(&c);
    if (data_out == nullptr)
        return false;

    auto mesh_in = mesh_in_slot_.CallAs<mesh::CallMesh>();
    if (mesh_in == nullptr)
        return false;

    auto parts_in = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto meta = data_out->getMetaData();
    auto mesh_meta = mesh_in->getMetaData();
    mesh_meta.m_frame_ID = meta.m_frame_ID;
    mesh_in->setMetaData(mesh_meta);
    if (!(*mesh_in)(1))
        return false;

    parts_in->SetFrameID(meta.m_frame_ID);
    if (!(*parts_in)(1))
        return false;

    mesh_meta = mesh_in->getMetaData();
    data_out->setMetaData(mesh_meta);

    return true;
}


bool megamol::thermodyn::MeshAddColor::assert_data(
    mesh::CallMesh& mesh, core::moldyn::MultiParticleDataCall& particles, core::view::CallGetTransferFunction* tf) {
    auto const pl_count = particles.GetParticleListCount();

    auto const meshes = mesh.getData()->accessMeshes();

    if (meshes.size() != pl_count)
        return false;

    colors_.resize(pl_count);

    mesh_col_ = std::make_shared<mesh::MeshDataAccessCollection>();

    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto& colors = colors_[pl_idx];
        auto const& parts = particles.AccessParticles(pl_idx);

        auto const mesh_ident = std::string("mesh_") + std::to_string(pl_idx);

        auto const fit = meshes.find(mesh_ident);
        if (fit == meshes.end())
            return false;

        auto const& mesh = fit->second;

        auto const idx_fit = std::find_if(mesh.attributes.begin(), mesh.attributes.end(), [](auto const& el) {
            return el.semantic == mesh::MeshDataAccessCollection::AttributeSemanticType::UNKNOWN;
        });

        if (idx_fit == mesh.attributes.end())
            return false;

        auto const idx_ptr = reinterpret_cast<uint32_t*>(idx_fit->data + idx_fit->offset);
        auto const idx_count = idx_fit->byte_size / idx_fit->stride;

        colors.clear();
        colors.reserve(idx_count);

        auto const cr_acc = parts.GetParticleStore().GetCRAcc();
        auto const cg_acc = parts.GetParticleStore().GetCGAcc();
        auto const cb_acc = parts.GetParticleStore().GetCBAcc();
        auto const ca_acc = parts.GetParticleStore().GetCAAcc();

        if ((parts.GetColourDataType() == core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I ||
                parts.GetColourDataType() == core::moldyn::SimpleSphericalParticles::COLDATA_DOUBLE_I) &&
            tf != nullptr) {
            auto const tf_tex = tf->GetTextureData();
            auto const tf_size = tf->TextureSize();

            auto const min_val = parts.GetMinColourIndexValue();
            auto const max_val = parts.GetMaxColourIndexValue();
            auto const fac = 1.0f / (max_val - min_val + 1e-8f);

            for (std::decay_t<decltype(idx_count)> i = 0; i < idx_count; ++i) {
                auto const idx = idx_ptr[i];
                auto const val = cr_acc->Get_f(idx);
                colors.push_back(stdplugin::datatools::get_sample_from_tf(tf_tex, tf_size, val, min_val, fac));
            }
        } else {
            for (std::decay_t<decltype(idx_count)> i = 0; i < idx_count; ++i) {
                auto const idx = idx_ptr[i];
                colors.push_back(
                    glm::vec4(cr_acc->Get_f(idx), cg_acc->Get_f(idx), cb_acc->Get_f(idx), ca_acc->Get_f(idx)));
            }
        }

        auto index_data = mesh.indices;
        auto attributes = mesh.attributes;
        attributes.push_back(
            {reinterpret_cast<decltype(mesh::MeshDataAccessCollection::VertexAttribute::data)>(colors.data()),
                colors.size() * sizeof(std::decay_t<decltype(colors)>::value_type), 4,
                mesh::MeshDataAccessCollection::ValueType::FLOAT, sizeof(std::decay_t<decltype(colors)>::value_type), 0,
                mesh::MeshDataAccessCollection::AttributeSemanticType::COLOR});

        mesh_col_->addMesh(mesh_ident, attributes, index_data);
    }

    return true;
}
