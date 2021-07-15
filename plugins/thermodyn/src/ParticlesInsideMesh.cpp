#include "ParticlesInsideMesh.h"


megamol::thermodyn::ParticlesInsideMesh::ParticlesInsideMesh()
        : data_out_slot_("dataOut", ""), mesh_in_slot_("meshIn", ""), parts_in_slot_("partsIn", "") {
    data_out_slot_.SetCallback<core::moldyn::MultiParticleDataCall, 0>(&ParticlesInsideMesh::get_data_cb);
    data_out_slot_.SetCallback<core::moldyn::MultiParticleDataCall, 1>(&ParticlesInsideMesh::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    mesh_in_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&mesh_in_slot_);

    parts_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&parts_in_slot_);
}


megamol::thermodyn::ParticlesInsideMesh::~ParticlesInsideMesh() {
    this->Release();
}


bool megamol::thermodyn::ParticlesInsideMesh::create() {
    return true;
}


void megamol::thermodyn::ParticlesInsideMesh::release() {}


bool megamol::thermodyn::ParticlesInsideMesh::get_data_cb(core::Call& c) {
    auto data_out = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (data_out == nullptr)
        return false;

    auto mesh_in = mesh_in_slot_.CallAs<mesh::CallMesh>();
    if (mesh_in == nullptr)
        return false;

    auto parts_in = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto const req_frame_id = data_out->FrameID();

    auto meta = mesh_in->getMetaData();
    meta.m_frame_ID = req_frame_id;
    mesh_in->setMetaData(meta);
    if (!(*mesh_in)(0))
        return false;

    parts_in->SetFrameID(req_frame_id);
    if (!(*parts_in)(0))
        return false;

    if (mesh_in->hasUpdate() || parts_in->DataHash() != in_data_hash_ || parts_in->FrameID() != frame_id_) {
        if (!assert_data(*mesh_in, *parts_in))
            return false;
        in_data_hash_ = parts_in->DataHash();
        frame_id_ = parts_in->FrameID();
        ++out_data_hash_;
    }

    *data_out = *parts_in;
    parts_in->SetUnlocker(nullptr, false);

    auto const pl_count = data_out->GetParticleListCount();
    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto& parts = data_out->AccessParticles(pl_idx);
        auto const& icol = icol_data_[pl_idx];
        parts.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, icol.data());
    }

    data_out->SetFrameID(frame_id_);
    data_out->SetDataHash(out_data_hash_);

    return true;
}


bool megamol::thermodyn::ParticlesInsideMesh::get_extent_cb(core::Call& c) {
    auto data_out = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (data_out == nullptr)
        return false;

    auto mesh_in = mesh_in_slot_.CallAs<mesh::CallMesh>();
    if (mesh_in == nullptr)
        return false;

    auto parts_in = parts_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto const req_frame_id = data_out->FrameID();

    auto meta = mesh_in->getMetaData();
    meta.m_frame_ID = req_frame_id;
    mesh_in->setMetaData(meta);
    if (!(*mesh_in)(1))
        return false;

    parts_in->SetFrameID(req_frame_id);
    if (!(*parts_in)(1))
        return false;

    data_out->SetFrameCount(parts_in->FrameCount());
    data_out->AccessBoundingBoxes() = parts_in->AccessBoundingBoxes();

    return true;
}


bool megamol::thermodyn::ParticlesInsideMesh::assert_data(
    mesh::CallMesh& mesh, core::moldyn::MultiParticleDataCall& particles) {
    auto const pl_count = particles.GetParticleListCount();

    auto const meshes = mesh.getData()->accessMeshes();

    if (meshes.size() != pl_count)
        return false;

    icol_data_.resize(pl_count);

    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& parts = particles.AccessParticles(pl_idx);
        auto& icol_data = icol_data_[pl_idx];

        auto const mesh_ident = std::string("mesh_") + std::to_string(pl_idx);

        auto const fit = meshes.find(mesh_ident);
        if (fit == meshes.end())
            return false;

        auto const& mesh = fit->second;

        auto const pos_fit = std::find_if(mesh.attributes.begin(), mesh.attributes.end(), [](auto const& el) {
            return el.semantic == mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION;
        });

        auto const pos_ptr = reinterpret_cast<glm::vec3*>(pos_fit->data + pos_fit->offset);
        auto const pos_count = pos_fit->byte_size / pos_fit->stride;

        auto const ind_ptr = reinterpret_cast<glm::uvec3*>(mesh.indices.data);
        auto const ind_count = mesh.indices.byte_size / sizeof(glm::uvec3);

        Polyhedron P;
        auto PF = PolyhedronFactory<HalfedgeDS>(pos_ptr, pos_count, ind_ptr, ind_count);
        P.delegate(PF);

        CGAL::Side_of_triangle_mesh<Polyhedron, Gt> inside(P);

        auto const p_count = parts.GetCount();

        auto const x_acc = parts.GetParticleStore().GetXAcc();
        auto const y_acc = parts.GetParticleStore().GetYAcc();
        auto const z_acc = parts.GetParticleStore().GetZAcc();

        icol_data.resize(p_count);

        for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
            auto const q_point = Point(x_acc->Get_f(p_idx), y_acc->Get_f(p_idx), z_acc->Get_f(p_idx));
            auto const res = inside(q_point);

            if (res == CGAL::ON_BOUNDARY || res == CGAL::ON_BOUNDED_SIDE) {
                icol_data[p_idx] = 1.0f;
            } else {
                icol_data[p_idx] = 0.0f;
            }
        }
    }

    return true;
}
