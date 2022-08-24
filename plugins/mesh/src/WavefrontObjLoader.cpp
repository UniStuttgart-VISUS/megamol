#include "WavefrontObjLoader.h"

#include "mmcore/param/FilePathParam.h"

megamol::mesh::WavefrontObjLoader::WavefrontObjLoader()
        : AbstractMeshDataSource()
        , m_version(0)
        , m_meta_data()
        , m_filename_slot("Wavefront OBJ filename", "The name of the obj file to load") {
    this->m_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_filename_slot);
}

megamol::mesh::WavefrontObjLoader::~WavefrontObjLoader() {}

bool megamol::mesh::WavefrontObjLoader::create(void) {
    AbstractMeshDataSource::create();
    return true;
}

bool megamol::mesh::WavefrontObjLoader::getMeshDataCallback(core::Call& caller) {
    CallMesh* lhs_mesh_call = dynamic_cast<CallMesh*>(&caller);
    CallMesh* rhs_mesh_call = m_mesh_rhs_slot.CallAs<CallMesh>();

    if (lhs_mesh_call == nullptr) {
        return false;
    }

    auto mesh_access_collections = std::make_shared<MeshDataAccessCollection>();

    // if there is a mesh connection to the right, pass on the mesh collection
    if (rhs_mesh_call != nullptr) {
        if (!(*rhs_mesh_call)(0)) {
            return false;
        }
        if (rhs_mesh_call->hasUpdate()) {
            ++m_version;
        }
        mesh_access_collections = rhs_mesh_call->getData();
    }

    if (this->m_filename_slot.IsDirty()) {
        m_filename_slot.ResetDirty();

        ++m_version;

        auto vislib_filename = m_filename_slot.Param<core::param::FilePathParam>()->Value();
        std::string filename(vislib_filename.generic_u8string());

        m_obj_model = std::make_shared<TinyObjModel>();

        std::string warn;
        std::string err;

        bool ret = tinyobj::LoadObj(
            &m_obj_model->attrib, &m_obj_model->shapes, &m_obj_model->materials, &warn, &err, filename.c_str());

        if (!warn.empty()) {
            std::cout << warn << std::endl;
        }

        if (!err.empty()) {
            std::cerr << err << std::endl;
        }

        if (!ret) {
            return false;
        }

        bool has_normals = !m_obj_model->attrib.normals.empty();
        bool has_texcoords = !(m_obj_model->attrib.texcoords.empty());

        // size_t vertex_cnt = m_obj_model->shapes.size() * 3;
        this->m_positions.clear();
        this->m_normals.clear();
        this->m_texcoords.clear();

        std::array<float, 6> bbox;
        bbox[0] = std::numeric_limits<float>::max();
        bbox[1] = std::numeric_limits<float>::max();
        bbox[2] = std::numeric_limits<float>::max();
        bbox[3] = -std::numeric_limits<float>::max();
        bbox[4] = -std::numeric_limits<float>::max();
        bbox[5] = -std::numeric_limits<float>::max();

        m_indices.resize(m_obj_model->shapes.size());
        m_positions.resize(m_obj_model->shapes.size());
        if (has_normals) {
            this->m_normals.resize(m_obj_model->shapes.size());
        }
        if (has_texcoords) {
            this->m_texcoords.resize(m_obj_model->shapes.size());
        }

        // Loop over shapes
        int index_offset = 0;
        int normal_offset = 0;
        for (size_t s = 0; s < m_obj_model->shapes.size(); s++) {
            auto& mesh = m_obj_model->shapes[s].mesh;
            auto& lines = m_obj_model->shapes[s].lines;

            has_normals = has_normals && !(mesh.indices[0].normal_index == -1);
            has_texcoords = has_texcoords && !(mesh.indices[0].texcoord_index == -1);

            uint64_t vertex_cnt = 0;
            if (!mesh.indices.empty()) {
                // Loop over faces(polygon)

                std::vector<unsigned int> index_vec;
                index_vec.reserve(mesh.indices.size());
                std::vector<unsigned int> normal_index_vec;
                normal_index_vec.reserve(mesh.indices.size());
                for (auto& id : mesh.indices) {
                    index_vec.emplace_back(id.vertex_index);
                    normal_index_vec.emplace_back(id.normal_index);
                }
                m_indices[s] = index_vec;
                std::sort(index_vec.begin(), index_vec.end());
                index_vec.erase(std::unique(index_vec.begin(), index_vec.end()), index_vec.end());
                std::sort(normal_index_vec.begin(), normal_index_vec.end());
                normal_index_vec.erase(
                    std::unique(normal_index_vec.begin(), normal_index_vec.end()), normal_index_vec.end());

                auto position_begin = &(m_obj_model->attrib.vertices[3 * index_offset]);
                m_positions[s].assign(position_begin, position_begin + 3*index_vec.size());

                if (has_normals) {
                    m_normals[s].resize(m_positions[s].size(), 0.0f);
                    for (auto& id: mesh.indices) {
                        m_normals[s][3 * (id.vertex_index - index_offset) + 0] += m_obj_model->attrib.normals[3 * id.normal_index + 0];
                        m_normals[s][3 * (id.vertex_index - index_offset) + 1] += m_obj_model->attrib.normals[3 * id.normal_index + 1];
                        m_normals[s][3 * (id.vertex_index - index_offset) + 2] += m_obj_model->attrib.normals[3 * id.normal_index + 2];
                    }
                    // normalize
                    for (int j = 0; j < m_normals[s].size()/3; j++) {
                        float const length = std::sqrtf(m_normals[s][3 * j] * m_normals[s][3 * j] +
                                                        m_normals[s][3 * j + 1] * m_normals[s][3 * j + 1] +
                                                        m_normals[s][3 * j + 2] * m_normals[s][3 * j +2]);
                        m_normals[s][3 * j + 0] = m_normals[s][3 * j + 0] / length;
                        m_normals[s][3 * j + 1] = m_normals[s][3 * j + 1] / length;
                        m_normals[s][3 * j + 2] = m_normals[s][3 * j + 2] / length;
                    }
                }
                if (has_texcoords) {
                    auto texcoords_begin = &(m_obj_model->attrib.texcoords[2 * index_offset]);
                    m_texcoords[s].assign(texcoords_begin, texcoords_begin + 2 * index_vec.size());
                }

                for (int j = 0; j < m_positions[s].size()/3; j++) {
                    bbox[0] = std::min(bbox[0], m_positions[s][3*j+0]);
                    bbox[1] = std::min(bbox[1], m_positions[s][3*j+1]);
                    bbox[2] = std::min(bbox[2], m_positions[s][3*j+2]);
                    bbox[3] = std::max(bbox[3], m_positions[s][3*j+0]);
                    bbox[4] = std::max(bbox[4], m_positions[s][3*j+1]);
                    bbox[5] = std::max(bbox[5], m_positions[s][3*j+2]);
                }
                index_offset += index_vec.size();
                normal_offset += normal_index_vec.size();

                vertex_cnt = m_positions[s].size() / 3;

            } else {
                // Loop over line
                size_t index_offset = 0;

                m_indices[s].resize(lines.num_line_vertices.size() * 2);
                m_positions[s].reserve(lines.num_line_vertices.size() * 3);

                for (size_t f = 0; f < lines.num_line_vertices.size(); f++) {
                    // access to vertex
                    tinyobj::index_t idx = lines.indices[index_offset];
                    tinyobj::real_t vx = m_obj_model->attrib.vertices[3 * idx.vertex_index + 0];
                    tinyobj::real_t vy = m_obj_model->attrib.vertices[3 * idx.vertex_index + 1];
                    tinyobj::real_t vz = m_obj_model->attrib.vertices[3 * idx.vertex_index + 2];

                    bbox[0] = std::min(bbox[0], static_cast<float>(vx));
                    bbox[1] = std::min(bbox[1], static_cast<float>(vy));
                    bbox[2] = std::min(bbox[2], static_cast<float>(vz));
                    bbox[3] = std::max(bbox[3], static_cast<float>(vx));
                    bbox[4] = std::max(bbox[4], static_cast<float>(vy));
                    bbox[5] = std::max(bbox[5], static_cast<float>(vz));

                    auto current_position_ptr = &(m_obj_model->attrib.vertices[3 * idx.vertex_index]);
                    m_positions[s].insert(m_positions[s].end(), current_position_ptr, current_position_ptr + 3);

                    // Loop over line segments
                    int fv = lines.num_line_vertices[f];
                    for (size_t v = 0; v < fv; v++) {
                        m_indices[s][index_offset + v] = lines.indices[index_offset + v].vertex_index;
                    }
                    index_offset += fv;
                    // per-face material
                    // m_obj_model->shapes[s].mesh.material_ids[f];
                }

                vertex_cnt = lines.num_line_vertices.size();
            }

            const auto pos_ptr = m_positions[s].data();
            std::vector<MeshDataAccessCollection::VertexAttribute> mesh_attributes;

            mesh_attributes.emplace_back(MeshDataAccessCollection::VertexAttribute{reinterpret_cast<uint8_t*>(pos_ptr),
                3 * vertex_cnt * MeshDataAccessCollection::getByteSize(MeshDataAccessCollection::FLOAT), 3,
                MeshDataAccessCollection::FLOAT, 12, 0, MeshDataAccessCollection::AttributeSemanticType::POSITION});

            if (has_normals) {
                const auto normal_ptr = m_normals[s].data();

                mesh_attributes.emplace_back(MeshDataAccessCollection::VertexAttribute{
                    reinterpret_cast<uint8_t*>(normal_ptr),
                    3 * vertex_cnt * MeshDataAccessCollection::getByteSize(MeshDataAccessCollection::FLOAT), 3,
                    MeshDataAccessCollection::FLOAT, 12, 0, MeshDataAccessCollection::AttributeSemanticType::NORMAL});
            }

            if (has_texcoords) {
                const auto texcoord_ptr = m_texcoords[s].data();
                mesh_attributes.emplace_back(MeshDataAccessCollection::VertexAttribute{
                    reinterpret_cast<uint8_t*>(texcoord_ptr),
                    2 * vertex_cnt * MeshDataAccessCollection::getByteSize(MeshDataAccessCollection::FLOAT), 2,
                    MeshDataAccessCollection::FLOAT, 8, 0, MeshDataAccessCollection::AttributeSemanticType::TEXCOORD});
            }

            MeshDataAccessCollection::IndexData mesh_indices;
            mesh_indices.data = reinterpret_cast<uint8_t*>(m_indices[s].data());
            mesh_indices.byte_size =
                m_indices[s].size() * MeshDataAccessCollection::getByteSize(MeshDataAccessCollection::UNSIGNED_INT);
            mesh_indices.type = MeshDataAccessCollection::UNSIGNED_INT;

            // TODO add file name?
            std::string identifier = m_obj_model->shapes[s].name;
            if (!mesh.indices.empty()) {
                m_mesh_access_collection.first->addMesh(
                    identifier, mesh_attributes, mesh_indices, MeshDataAccessCollection::PrimitiveType::TRIANGLES);
            } else {
                m_mesh_access_collection.first->addMesh(identifier, mesh_attributes, mesh_indices);
            }
            m_mesh_access_collection.second.push_back(identifier);
        }

        m_meta_data.m_bboxs.SetBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        m_meta_data.m_bboxs.SetClipBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        m_meta_data.m_frame_cnt = 1;
    }

    if (lhs_mesh_call->version() < m_version) {
        lhs_mesh_call->setMetaData(m_meta_data);
        //TODO fix bounding box / meta data handling
    }

    mesh_access_collections->append(*m_mesh_access_collection.first);
    lhs_mesh_call->setData(mesh_access_collections, m_version);

    return true;
}

bool megamol::mesh::WavefrontObjLoader::getMeshMetaDataCallback(core::Call& caller) {
    return AbstractMeshDataSource::getMeshMetaDataCallback(caller);
}

void megamol::mesh::WavefrontObjLoader::release() {}
