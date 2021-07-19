#include "ParticleSurface2.h"

#include "mmcore/param/FloatParam.h"


megamol::thermodyn::ParticleSurface2::ParticleSurface2()
        : data_out_slot_("dataOut", "")
        , data_in_slot_("dataIn", "")
        , flags_read_slot_("flagsRead", "")
        , alpha_slot_("alpha", "") {
    data_out_slot_.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &ParticleSurface2::get_data_cb);
    data_out_slot_.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &ParticleSurface2::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    flags_read_slot_.SetCompatibleCall<core::FlagCallRead_CPUDescription>();
    MakeSlotAvailable(&flags_read_slot_);

    alpha_slot_ << new core::param::FloatParam(2.5f, std::numeric_limits<float>::min());
    MakeSlotAvailable(&alpha_slot_);
}


megamol::thermodyn::ParticleSurface2::~ParticleSurface2() {
    this->Release();
}


bool megamol::thermodyn::ParticleSurface2::create() {
    return true;
}


void megamol::thermodyn::ParticleSurface2::release() {}


bool megamol::thermodyn::ParticleSurface2::get_data_cb(core::Call& c) {
    auto mesh_out = dynamic_cast<mesh::CallMesh*>(&c);
    if (mesh_out == nullptr)
        return false;

    auto data_in = data_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (data_in == nullptr)
        return false;

    auto flags_in = flags_read_slot_.CallAs<core::FlagCallRead_CPU>();

    auto meta = mesh_out->getMetaData();
    data_in->SetFrameID(meta.m_frame_ID);
    if (!(*data_in)(0))
        return false;

    if (flags_in) {
        (*flags_in)();
    }

    if (data_in->DataHash() != in_data_hash_ || data_in->FrameID() != frame_id_ || is_dirty() ||
        (flags_in != nullptr && flags_in->hasUpdate())) {
        if (!assert_data(*data_in, flags_in))
            return false;
        in_data_hash_ = data_in->DataHash();
        frame_id_ = data_in->FrameID();
        reset_dirty();
        ++version;
    }

    meta.m_frame_ID = frame_id_;
    mesh_out->setData(mesh_col_, version);
    mesh_out->setMetaData(meta);

    return true;
}


bool megamol::thermodyn::ParticleSurface2::get_extent_cb(core::Call& c) {
    auto mesh_out = dynamic_cast<mesh::CallMesh*>(&c);
    if (mesh_out == nullptr)
        return false;

    auto data_in = data_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (data_in == nullptr)
        return false;

    auto meta = mesh_out->getMetaData();
    data_in->SetFrameID(meta.m_frame_ID);
    if (!(*data_in)(1))
        return false;

    meta.m_frame_cnt = data_in->FrameCount();
    meta.m_bboxs.SetBoundingBox(data_in->AccessBoundingBoxes().ObjectSpaceBBox());
    meta.m_bboxs.SetClipBox(data_in->AccessBoundingBoxes().ObjectSpaceClipBox());

    mesh_out->setMetaData(meta);

    return true;
}


void conditionally_add_vertex(megamol::thermodyn::ParticleSurface2::vertex_con_t& vertices,
    megamol::thermodyn::ParticleSurface2::normals_con_t& normals,
    megamol::thermodyn::ParticleSurface2::index_con_t& indices,
    megamol::thermodyn::ParticleSurface2::index_con_t& part_indices,
    megamol::thermodyn::ParticleSurface2::idx_map_t& added_vertices,
    megamol::thermodyn::ParticleSurface2::Point_3 const& p, uint32_t idx, CGAL::Vector_3<CGAL::Epick> const& normal) {
    auto const fit = added_vertices.find(idx);
    if (fit == added_vertices.end()) {
        added_vertices[idx] = vertices.size();
        indices.push_back(vertices.size());
        vertices.push_back(glm::vec3(p.x(), p.y(), p.z()));
        normals.push_back(glm::vec3(normal.x(), normal.y(), normal.z()));
        part_indices.push_back(idx);
    } else {
        indices.push_back(fit->second);
    }
}


bool megamol::thermodyn::ParticleSurface2::assert_data(
    core::moldyn::MultiParticleDataCall& particles, core::FlagCallRead_CPU* flags) {
    auto const pl_count = particles.GetParticleListCount();

    auto const alpha = alpha_slot_.Param<core::param::FloatParam>()->Value();

    mesh_col_ = std::make_shared<mesh::MeshDataAccessCollection>();

    vertices_.resize(pl_count);
    normals_.resize(pl_count);
    indices_.resize(pl_count);
    part_indices_.resize(pl_count);

    uint64_t flag_idx = 0;
    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& parts = particles.AccessParticles(pl_idx);

        auto const x_acc = parts.GetParticleStore().GetXAcc();
        auto const y_acc = parts.GetParticleStore().GetYAcc();
        auto const z_acc = parts.GetParticleStore().GetZAcc();

        auto const p_count = parts.GetCount();

        auto& vertices = vertices_[pl_idx];
        vertices.clear();
        vertices.reserve(p_count);
        auto& normals = normals_[pl_idx];
        normals.clear();
        normals.reserve(p_count);
        auto& indices = indices_[pl_idx];
        indices.clear();
        indices.reserve(p_count);
        auto& part_indices = part_indices_[pl_idx];
        part_indices.clear();
        part_indices.reserve(p_count);

        std::list<point_w_info_t> points;

        if (flags) {
            auto const flags_data = *(flags->getData()->flags);
            for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx, ++flag_idx) {
                if (flags_data[flag_idx] != core::FlagStorage::FILTERED) {
                    points.push_back(
                        std::make_pair(Point_3(x_acc->Get_f(p_idx), y_acc->Get_f(p_idx), z_acc->Get_f(p_idx)), p_idx));
                }
            }
        } else {
            for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                points.push_back(
                    std::make_pair(Point_3(x_acc->Get_f(p_idx), y_acc->Get_f(p_idx), z_acc->Get_f(p_idx)), p_idx));
            }
        }

        Triangulation_3 tri = Triangulation_3(points.begin(), points.end());
        auto const as = Alpha_shape_3(tri, alpha);

        std::list<Facet> facets;
        as.get_alpha_shape_facets(std::back_inserter(facets), Alpha_shape_3::REGULAR);

        idx_map_t added_vertices;

        // https://stackoverflow.com/questions/15905833/saving-cgal-alpha-shape-surface-mesh
        for (auto const& face_from_list : facets) {
            auto face = face_from_list;
            if (as.classify(face.first) != Alpha_shape_3::EXTERIOR)
                face = as.mirror_facet(face);

            int idx[3] = {
                (face.second + 1) % 4,
                (face.second + 2) % 4,
                (face.second + 3) % 4,
            };

            if (face.second % 2 == 0)
                std::swap(idx[0], idx[1]);

            auto const& a = (face.first->vertex(idx[0])->point());
            auto const& b = (face.first->vertex(idx[1])->point());
            auto const& c = (face.first->vertex(idx[2])->point());

            auto const a_idx = face.first->vertex(idx[0])->info();
            auto const b_idx = face.first->vertex(idx[1])->info();
            auto const c_idx = face.first->vertex(idx[2])->info();

            auto normal = CGAL::normal(a, b, c);
            auto const length = std::sqrtf(normal.squared_length());
            normal /= length;

            conditionally_add_vertex(vertices, normals, indices, part_indices, added_vertices, a, a_idx, normal);
            conditionally_add_vertex(vertices, normals, indices, part_indices, added_vertices, b, b_idx, normal);
            conditionally_add_vertex(vertices, normals, indices, part_indices, added_vertices, c, c_idx, normal);
        }

        mesh::MeshDataAccessCollection::IndexData index_data;
        index_data.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
        index_data.byte_size = indices.size() * sizeof(std::decay_t<decltype(indices)>::value_type);
        index_data.data = reinterpret_cast<decltype(index_data.data)>(indices.data());

        std::vector<mesh::MeshDataAccessCollection::VertexAttribute> attributes = {
            {reinterpret_cast<decltype(mesh::MeshDataAccessCollection::VertexAttribute::data)>(vertices.data()),
                vertices.size() * sizeof(std::decay_t<decltype(vertices)>::value_type), 3,
                mesh::MeshDataAccessCollection::ValueType::FLOAT, sizeof(std::decay_t<decltype(vertices)>::value_type),
                0, mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION},
            {reinterpret_cast<decltype(mesh::MeshDataAccessCollection::VertexAttribute::data)>(normals.data()),
                normals.size() * sizeof(std::decay_t<decltype(normals)>::value_type), 3,
                mesh::MeshDataAccessCollection::ValueType::FLOAT, sizeof(std::decay_t<decltype(normals)>::value_type),
                0, mesh::MeshDataAccessCollection::AttributeSemanticType::NORMAL},
            {reinterpret_cast<decltype(mesh::MeshDataAccessCollection::VertexAttribute::data)>(part_indices.data()),
                part_indices.size() * sizeof(std::decay_t<decltype(part_indices)>::value_type), 1,
                mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT,
                sizeof(std::decay_t<decltype(part_indices)>::value_type), 0,
                mesh::MeshDataAccessCollection::AttributeSemanticType::UNKNOWN}};

        mesh_col_->addMesh(std::string("mesh_") + std::to_string(pl_idx), attributes, index_data);
    }

    return true;
}
