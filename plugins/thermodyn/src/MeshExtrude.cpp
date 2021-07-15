#include "MeshExtrude.h"

#include "mmcore/param/FloatParam.h"


megamol::thermodyn::MeshExtrude::MeshExtrude()
        : data_out_slot_("dataOut", "")
        , mesh_in_slot_("meshIn", "")
        , parts_in_slot_("partsIn", "")
        , crit_temp_slot_("Tc", "") {
    data_out_slot_.SetCallback<mesh::CallMesh, 0>(&MeshExtrude::get_data_cb);
    data_out_slot_.SetCallback<mesh::CallMesh, 1>(&MeshExtrude::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    mesh_in_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&mesh_in_slot_);

    parts_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&parts_in_slot_);

    crit_temp_slot_ << new core::param::FloatParam(1.7f);
    MakeSlotAvailable(&crit_temp_slot_);
}


megamol::thermodyn::MeshExtrude::~MeshExtrude() {
    this->Release();
}


bool megamol::thermodyn::MeshExtrude::create() {
    return true;
}


void megamol::thermodyn::MeshExtrude::release() {}


bool megamol::thermodyn::MeshExtrude::get_data_cb(core::Call& c) {
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
    if (!(*mesh_in)(0))
        return false;

    parts_in->SetFrameID(meta.m_frame_ID);
    if (!(*parts_in)(0))
        return false;

    if (mesh_in->hasUpdate() || parts_in->DataHash() != in_data_hash_ || parts_in->FrameID() != frame_id_ ||
        is_dirty()) {
        if (!assert_data(*mesh_in, *parts_in))
            return false;
        in_data_hash_ = parts_in->DataHash();
        frame_id_ = parts_in->FrameID();
        ++version;
        reset_dirty();
    }

    meta.m_frame_ID = frame_id_;
    data_out->setMetaData(meta);
    data_out->setData(mesh_col_, version);

    return true;
}


bool megamol::thermodyn::MeshExtrude::get_extent_cb(core::Call& c) {
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


bool megamol::thermodyn::MeshExtrude::assert_data(
    mesh::CallMesh& mesh, core::moldyn::MultiParticleDataCall& particles) {
    auto const thickness = [](float T, float T_c) -> float {
        return -1.720f * std::powf((T_c - T) / T_c, 1.89f) + 1.103f * std::powf((T_c - T) / T_c, -0.62f);
    };

    auto const tc = crit_temp_slot_.Param<core::param::FloatParam>()->Value();

    auto const pl_count = particles.GetParticleListCount();

    auto const meshes = mesh.getData()->accessMeshes();

    if (meshes.size() != pl_count)
        return false;

    mesh_col_ = std::make_shared<mesh::MeshDataAccessCollection>();

    vertices_.resize(pl_count);

    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& parts = particles.AccessParticles(pl_idx);
        if (parts.GetColourDataType() != core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I &&
            parts.GetColourDataType() != core::moldyn::SimpleSphericalParticles::COLDATA_DOUBLE_I)
            return false;

        auto& vertices = vertices_[pl_idx];

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

        auto const pos_fit = std::find_if(mesh.attributes.begin(), mesh.attributes.end(), [](auto const& el) {
            return el.semantic == mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION;
        });

        auto const pos_ptr = reinterpret_cast<glm::vec3*>(pos_fit->data + pos_fit->offset);
        auto const pos_count = pos_fit->byte_size / pos_fit->stride;

        vertices.resize(pos_count);
        std::copy(pos_ptr, pos_ptr + pos_count, vertices.begin());

        auto const normal_fit = std::find_if(mesh.attributes.begin(), mesh.attributes.end(), [](auto const& el) {
            return el.semantic == mesh::MeshDataAccessCollection::AttributeSemanticType::NORMAL;
        });

        auto const normal_ptr = reinterpret_cast<glm::vec3*>(normal_fit->data + normal_fit->offset);
        auto const normal_count = normal_fit->byte_size / normal_fit->stride;

        auto const i_acc = parts.GetParticleStore().GetCRAcc();

        for (std::decay_t<decltype(idx_count)> i = 0; i < idx_count; ++i) {
            auto const idx = idx_ptr[i];

            auto const val = i_acc->Get_f(idx);

            auto const normal = normal_ptr[i];
            auto& pos = vertices[i];

            auto const D = thickness(val, tc);

            pos = pos + (D * normal);
        }

        auto index_data = mesh.indices;
        auto attributes = mesh.attributes;

        {
            auto const pos_fit = std::find_if(attributes.begin(), attributes.end(), [](auto const& el) {
                return el.semantic == mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION;
            });

            attributes.erase(pos_fit);
        }

        attributes.push_back(
            {reinterpret_cast<decltype(mesh::MeshDataAccessCollection::VertexAttribute::data)>(vertices.data()),
                vertices.size() * sizeof(std::decay_t<decltype(vertices)>::value_type), 3,
                mesh::MeshDataAccessCollection::ValueType::FLOAT, sizeof(std::decay_t<decltype(vertices)>::value_type),
                0, mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION});

        mesh_col_->addMesh(mesh_ident, attributes, index_data);
    }

    return true;
}
