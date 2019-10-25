#include "stdafx.h"

#include "WavefrontObjLoader.h"

#include "mmcore/param/FilePathParam.h"

megamol::mesh::WavefrontObjLoader::WavefrontObjLoader()
    : core::Module()
    , m_mesh_data_access(std::make_shared<MeshDataAccessCollection>())
    , m_meta_data()
    , m_filename_slot("Wavefront OBJ filename", "The name of the obj file to load")
    , m_getData_slot("CallMesh", "The slot publishing the loaded data") {
    this->m_getData_slot.SetCallback(CallMesh::ClassName(), "GetData", &WavefrontObjLoader::getDataCallback);
    this->m_getData_slot.SetCallback(CallMesh::ClassName(), "GetMetaData", &WavefrontObjLoader::getDataCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    this->m_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_filename_slot);
}

megamol::mesh::WavefrontObjLoader::~WavefrontObjLoader() { this->Release(); }

bool megamol::mesh::WavefrontObjLoader::create(void) { return true; }

bool megamol::mesh::WavefrontObjLoader::getDataCallback(core::Call& caller) {

    CallMesh* cm = dynamic_cast<CallMesh*>(&caller);

    // TODO detect whether a mesh data collection is given by lhs caller (chaining)

    if (cm == NULL) return false;

    if (this->m_filename_slot.IsDirty()) {
        m_filename_slot.ResetDirty();

        auto vislib_filename = m_filename_slot.Param<core::param::FilePathParam>()->Value();
        std::string filename(vislib_filename.PeekBuffer());

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

        bool has_normals = ! m_obj_model->attrib.normals.empty();
        bool has_texcoords = ! m_obj_model->attrib.texcoords.empty();

        size_t vertex_cnt = m_obj_model->shapes.size() * 3;
        this->m_positions.clear();
        this->m_normals.clear();
        this->m_texcoords.clear();

        this->m_positions.reserve(vertex_cnt);

        if (has_normals) {
            this->m_normals.reserve(vertex_cnt);
        }

        if (has_texcoords) {
            this->m_texcoords.reserve(vertex_cnt);
        }

        std::array<float, 6> bbox;

        bbox[0] = std::numeric_limits<float>::max();
        bbox[1] = std::numeric_limits<float>::max();
        bbox[2] = std::numeric_limits<float>::max();
        bbox[3] = std::numeric_limits<float>::min();
        bbox[4] = std::numeric_limits<float>::min();
        bbox[5] = std::numeric_limits<float>::min();

        // Loop over shapes
        for (size_t s = 0; s < m_obj_model->shapes.size(); s++) {
            // Loop over faces(polygon)
            size_t index_offset = 0;
            for (size_t f = 0; f < m_obj_model->shapes[s].mesh.num_face_vertices.size(); f++) {
                int fv = m_obj_model->shapes[s].mesh.num_face_vertices[f];

                assert(fv == 3); // assume that triangulation was forced

                // Loop over vertices in the face.
                for (size_t v = 0; v < fv; v++) {
                    // access to vertex
                    tinyobj::index_t idx = m_obj_model->shapes[s].mesh.indices[index_offset + v];
                    tinyobj::real_t vx = m_obj_model->attrib.vertices[3 * idx.vertex_index + 0];
                    tinyobj::real_t vy = m_obj_model->attrib.vertices[3 * idx.vertex_index + 1];
                    tinyobj::real_t vz = m_obj_model->attrib.vertices[3 * idx.vertex_index + 2];
                    
                    bbox[0] = std::min(bbox[0], static_cast<float>(vx));
                    bbox[1] = std::min(bbox[1], static_cast<float>(vy));
                    bbox[2] = std::min(bbox[2], static_cast<float>(vz));
                    bbox[3] = std::max(bbox[3], static_cast<float>(vx));
                    bbox[4] = std::max(bbox[4], static_cast<float>(vy));
                    bbox[5] = std::max(bbox[5], static_cast<float>(vz));

                    if (has_normals) {
                        tinyobj::real_t nx = m_obj_model->attrib.normals[3 * idx.normal_index + 0];
                        tinyobj::real_t ny = m_obj_model->attrib.normals[3 * idx.normal_index + 1];
                        tinyobj::real_t nz = m_obj_model->attrib.normals[3 * idx.normal_index + 2];
                        this->m_normals.insert(m_normals.end(), {nx, ny, nz});
                    }

                    if (has_texcoords) {
                        tinyobj::real_t tx = m_obj_model->attrib.texcoords[2 * idx.texcoord_index + 0];
                        tinyobj::real_t ty = m_obj_model->attrib.texcoords[2 * idx.texcoord_index + 1];
                        this->m_texcoords.insert(m_texcoords.end(), {tx, ty});
                    }

                    this->m_positions.insert(m_positions.end(), {vx, vy, vz});

                    this->m_indices.emplace_back(index_offset + v);
                }
                index_offset += fv;

                // per-face material
                // m_obj_model->shapes[s].mesh.material_ids[f];
            }
        }

        std::vector<MeshDataAccessCollection::VertexAttribute> mesh_attributes;

        mesh_attributes.emplace_back(
            MeshDataAccessCollection::VertexAttribute{reinterpret_cast<uint8_t*>(this->m_positions.data()),
                m_positions.size() * MeshDataAccessCollection::getByteSize(MeshDataAccessCollection::FLOAT), 3,
                MeshDataAccessCollection::FLOAT, 0, 0});

        if (has_normals){
            mesh_attributes.emplace_back(
                MeshDataAccessCollection::VertexAttribute{reinterpret_cast<uint8_t*>(this->m_normals.data()),
                    m_normals.size() * MeshDataAccessCollection::getByteSize(MeshDataAccessCollection::FLOAT), 3,
                    MeshDataAccessCollection::FLOAT, 0, 0});
        }

        if (has_texcoords){
            mesh_attributes.emplace_back(
                MeshDataAccessCollection::VertexAttribute{reinterpret_cast<uint8_t*>(this->m_texcoords.data()),
                    m_texcoords.size() * MeshDataAccessCollection::getByteSize(MeshDataAccessCollection::FLOAT), 2,
                    MeshDataAccessCollection::FLOAT, 0, 0});
        }

        MeshDataAccessCollection::IndexData mesh_indices;
        mesh_indices.data = reinterpret_cast<uint8_t*>(m_indices.data());
        mesh_indices.byte_size = m_indices.size() * MeshDataAccessCollection::getByteSize(MeshDataAccessCollection::UNSIGNED_INT);
        mesh_indices.type = MeshDataAccessCollection::UNSIGNED_INT;

        this->m_mesh_data_access->addMesh(mesh_attributes, mesh_indices);

        ++(m_meta_data.m_data_hash);

        m_meta_data.m_bboxs.SetObjectSpaceBBox(
            bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        m_meta_data.m_bboxs.SetObjectSpaceClipBox(
            bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);

        m_meta_data.m_frame_cnt = 1;
        m_meta_data.m_frame_ID = 0;
    }

    cm->setMetaData(m_meta_data);
    cm->setData(m_mesh_data_access);
}

bool megamol::mesh::WavefrontObjLoader::getMetaDataCallback(core::Call& caller) { 

    CallMesh* cm = dynamic_cast<CallMesh*>(&caller);

    if (cm == NULL) return false;

    cm->setMetaData(m_meta_data);

    return true;
}

void megamol::mesh::WavefrontObjLoader::release() {}
