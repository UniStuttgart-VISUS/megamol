#include "MeshToParticles.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "mesh/MeshDataAccessor.h"


megamol::mesh::MeshToParticles::MeshToParticles() : data_out_slot_("dataOut", ""), data_in_slot_("dataIn", "") {
    data_out_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &MeshToParticles::get_data_cb);
    data_out_slot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &MeshToParticles::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<CallMeshDescription>();
    MakeSlotAvailable(&data_in_slot_);
}


megamol::mesh::MeshToParticles::~MeshToParticles() {
    this->Release();
}


bool megamol::mesh::MeshToParticles::create() {
    return true;
}


void megamol::mesh::MeshToParticles::release() {}


bool megamol::mesh::MeshToParticles::get_data_cb(core::Call& c) {
    auto parts_out = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (parts_out == nullptr)
        return false;

    auto mesh_in = data_in_slot_.CallAs<mesh::CallMesh>();
    if (mesh_in == nullptr)
        return false;

    auto const req_frame_id = parts_out->FrameID();
    auto meta = mesh_in->getMetaData();
    meta.m_frame_ID = req_frame_id;
    mesh_in->setMetaData(meta);
    if (!(*mesh_in)(0))
        return false;

    if (mesh_in->hasUpdate()) {
        if (!assert_data(*mesh_in))
            return false;
        meta = mesh_in->getMetaData();
        in_frame_id = meta.m_frame_ID;
        ++out_data_hash_;
    }

    parts_out->SetParticleListCount(data_.size());
    for (decltype(data_)::size_type i = 0; i < data_.size(); ++i) {
        auto const& el = data_[i];
        auto& parts = parts_out->AccessParticles(i);
        auto const p_count = el.size() / 3;
        parts.SetCount(p_count);
        parts.SetVertexData(core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, el.data());
    }
    parts_out->SetDataHash(out_data_hash_);
    parts_out->SetFrameID(in_frame_id);

    return true;
}


bool megamol::mesh::MeshToParticles::get_extent_cb(core::Call& c) {
    auto parts_out = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (parts_out == nullptr)
        return false;

    auto mesh_in = data_in_slot_.CallAs<mesh::CallMesh>();
    if (mesh_in == nullptr)
        return false;

    auto const req_frame_id = parts_out->FrameID();
    auto meta = mesh_in->getMetaData();
    meta.m_frame_ID = req_frame_id;
    mesh_in->setMetaData(meta);
    if (!(*mesh_in)(1))
        return false;
    meta = mesh_in->getMetaData();

    parts_out->SetFrameCount(meta.m_frame_cnt);
    parts_out->AccessBoundingBoxes().SetObjectSpaceBBox(meta.m_bboxs.BoundingBox());
    parts_out->AccessBoundingBoxes().SetObjectSpaceClipBox(meta.m_bboxs.ClipBox());

    return true;
}


bool megamol::mesh::MeshToParticles::assert_data(mesh::CallMesh& meshes) {
    auto const meshes_ptr = meshes.getData();
    auto const& meshes_data = meshes_ptr->accessMeshes();

    data_.resize(meshes_data.size());

    auto m_it = meshes_data.begin();
    for (std::decay_t<decltype(meshes_data)>::size_type m_idx = 0; m_idx < meshes_data.size(); ++m_idx, ++m_it) {
        auto& part_data = data_[m_idx];

        auto const& mesh_data = m_it->second;

        /*auto mesh_acc = MeshDataTriangleAccessor(mesh_data);
        auto const m_el_count = mesh_acc.GetCount();

        for (std::decay_t<decltype(m_el_count)> el_idx = 0; el_idx < m_el_count; ++el_idx) {
            auto const pos = mesh_acc.GetPosition(el_idx);
        }*/

        auto const fit = std::find_if(mesh_data.attributes.begin(), mesh_data.attributes.end(),
            [](auto const& el) { return el.semantic == MeshDataAccessCollection::AttributeSemanticType::POSITION; });

        if (fit == mesh_data.attributes.end())
            continue;

        if (fit->stride != sizeof(glm::vec3))
            continue;

        auto el_count = fit->byte_size / sizeof(glm::vec3);

        part_data.resize(el_count * 3);

        std::copy(fit->data + fit->offset, fit->data + fit->offset + fit->byte_size, reinterpret_cast<uint8_t*>(part_data.data()));
    }

    return true;
}
