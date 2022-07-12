#include "glTFFileLoader.h"

#include "mmcore/param/FilePathParam.h"

#ifndef TINYGLTF_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#endif // !TINYGLTF_IMPLEMENTATION
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif // !STB_IMAGE_IMPLEMENTATION
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif // !STB_IMAGE_WRITE_IMPLEMENTATION

#include "tiny_gltf.h"

megamol::mesh::GlTFFileLoader::GlTFFileLoader()
        : AbstractMeshDataSource()
        , m_version(0)
        , m_glTFFilename_slot("glTF filename", "The name of the gltf file to load")
        , m_gltf_slot("gltfModels", "The slot publishing the loaded data") {
    this->m_gltf_slot.SetCallback(CallGlTFData::ClassName(), "GetData", &GlTFFileLoader::getGltfDataCallback);
    this->m_gltf_slot.SetCallback(CallGlTFData::ClassName(), "GetMetaData", &GlTFFileLoader::getGltfMetaDataCallback);
    this->m_gltf_slot.SetCallback(CallGlTFData::ClassName(), "Disconnect", &GlTFFileLoader::disconnectGltfCallback);
    this->MakeSlotAvailable(&this->m_gltf_slot);

    this->m_glTFFilename_slot << new core::param::FilePathParam(
        "", core::param::FilePathParam::Flag_File_RestrictExtension, {"gltf"});
    this->MakeSlotAvailable(&this->m_glTFFilename_slot);
}

megamol::mesh::GlTFFileLoader::~GlTFFileLoader() {
    this->Release();
}

bool megamol::mesh::GlTFFileLoader::getGltfDataCallback(core::Call& caller) {
    CallGlTFData* gltf_call = dynamic_cast<CallGlTFData*>(&caller);
    if (gltf_call == NULL)
        return false;

    auto has_update = checkAndLoadGltfModel();
    if (has_update) {
        ++m_version;
    }

    if (gltf_call->version() < m_version) {
        gltf_call->setData(
            {m_glTFFilename_slot.Param<core::param::FilePathParam>()->Value().generic_u8string(), m_gltf_model},
            m_version);
    }

    return true;
}

bool megamol::mesh::GlTFFileLoader::getGltfMetaDataCallback(core::Call& caller) {
    return true;
}

bool megamol::mesh::GlTFFileLoader::getMeshDataCallback(core::Call& caller) {

    CallMesh* lhs_mesh_call = dynamic_cast<CallMesh*>(&caller);
    CallMesh* rhs_mesh_call = m_mesh_rhs_slot.CallAs<CallMesh>();

    if (lhs_mesh_call == NULL) {
        return false;
    }

    syncMeshAccessCollection(lhs_mesh_call, rhs_mesh_call);

    // if there is a mesh connection to the right, pass on the mesh collection
    if (rhs_mesh_call != NULL) {
        if (!(*rhs_mesh_call)(0)) {
            return false;
        }
        if (rhs_mesh_call->hasUpdate()) {
            ++m_version;
            rhs_mesh_call->getData();
        }
    }

    auto has_update = checkAndLoadGltfModel();
    if (has_update) {
        ++m_version;
    }

    if (lhs_mesh_call->version() < m_version) {
        for (auto const& identifier : m_mesh_access_collection.second) {
            m_mesh_access_collection.first->deleteMesh(identifier);
        }
        m_mesh_access_collection.second.clear();

        // set data and version to signal update
        lhs_mesh_call->setData(m_mesh_access_collection.first, m_version);

        // compute mesh call specific update
        std::array<float, 6> bbox;

        bbox[0] = std::numeric_limits<float>::max();
        bbox[1] = std::numeric_limits<float>::max();
        bbox[2] = std::numeric_limits<float>::max();
        bbox[3] = std::numeric_limits<float>::lowest();
        bbox[4] = std::numeric_limits<float>::lowest();
        bbox[5] = std::numeric_limits<float>::lowest();

        auto model = m_gltf_model;

        if (model == nullptr)
            return false;

        for (size_t mesh_idx = 0; mesh_idx < model->meshes.size(); mesh_idx++) {

            auto primitive_cnt = model->meshes[mesh_idx].primitives.size();

            for (size_t primitive_idx = 0; primitive_idx < primitive_cnt; ++primitive_idx) {

                std::vector<MeshDataAccessCollection::VertexAttribute> mesh_attributes;
                MeshDataAccessCollection::IndexData mesh_indices;

                auto& indices_accessor = model->accessors[model->meshes[mesh_idx].primitives[primitive_idx].indices];
                auto& indices_bufferView = model->bufferViews[indices_accessor.bufferView];
                auto& indices_buffer = model->buffers[indices_bufferView.buffer];

                mesh_indices.byte_size = (indices_accessor.count * indices_accessor.ByteStride(indices_bufferView));
                mesh_indices.data = reinterpret_cast<uint8_t*>(
                    indices_buffer.data.data() + indices_bufferView.byteOffset + indices_accessor.byteOffset);
                mesh_indices.type = MeshDataAccessCollection::covertToValueType(indices_accessor.componentType);

                auto& vertex_attributes = model->meshes[mesh_idx].primitives[primitive_idx].attributes;
                for (auto attrib : vertex_attributes) {
                    auto& vertexAttrib_accessor = model->accessors[attrib.second];
                    auto& vertexAttrib_bufferView = model->bufferViews[vertexAttrib_accessor.bufferView];
                    auto& vertexAttrib_buffer = model->buffers[vertexAttrib_bufferView.buffer];

                    MeshDataAccessCollection::AttributeSemanticType attrib_semantic;

                    if (attrib.first == "POSITION") {
                        attrib_semantic = MeshDataAccessCollection::AttributeSemanticType::POSITION;
                    } else if (attrib.first == "NORMAL") {
                        attrib_semantic = MeshDataAccessCollection::AttributeSemanticType::NORMAL;
                    } else if (attrib.first == "TANGENT") {
                        attrib_semantic = MeshDataAccessCollection::AttributeSemanticType::TANGENT;
                    } else if (attrib.first == "TEXCOORD_0") {
                        attrib_semantic = MeshDataAccessCollection::AttributeSemanticType::TEXCOORD;
                    } else if (attrib.first == "COLOR_0") {
                        attrib_semantic = MeshDataAccessCollection::AttributeSemanticType::COLOR;
                    } else {
                        attrib_semantic = MeshDataAccessCollection::AttributeSemanticType::UNKNOWN;
                    }


                    uint8_t* data_ptr = nullptr;
                    size_t attrib_byte_stride = vertexAttrib_accessor.ByteStride(vertexAttrib_bufferView);
                    size_t attrib_byte_offset = 0;
                    // check bufferView stride for 0 to detect interleaved data
                    if (vertexAttrib_bufferView.byteStride == 0) {
                        // if interleaved, do not apply accessor byte offset to data pointer
                        data_ptr = reinterpret_cast<uint8_t*>(
                            vertexAttrib_buffer.data.data() + vertexAttrib_bufferView.byteOffset);
                        // and instead use attribute offset
                        attrib_byte_offset = vertexAttrib_accessor.byteOffset;
                    } else {
                        // if non-interleaved, apply accessor byte offset to data pointer
                        data_ptr = reinterpret_cast<uint8_t*>(vertexAttrib_buffer.data.data() +
                                                              vertexAttrib_bufferView.byteOffset +
                                                              vertexAttrib_accessor.byteOffset);
                        // and do not set any attribute offset because offset > 0 with non-interleaved
                        // attributes suggests a (VVVNNNCCC) layout which we will use as a
                        // (VVV)(NNN)(CCC) layout instead
                    }

                    mesh_attributes.emplace_back(MeshDataAccessCollection::VertexAttribute{data_ptr,
                        (vertexAttrib_accessor.count * attrib_byte_stride),
                        static_cast<unsigned int>(vertexAttrib_accessor.type),
                        MeshDataAccessCollection::covertToValueType(vertexAttrib_accessor.componentType),
                        attrib_byte_stride, attrib_byte_offset, attrib_semantic});
                }

                std::string identifier =
                    m_glTFFilename_slot.Param<core::param::FilePathParam>()->Value().generic_u8string() +
                    model->meshes[mesh_idx].name + "_" + std::to_string(primitive_idx);
                m_mesh_access_collection.first->addMesh(identifier, mesh_attributes, mesh_indices);
                m_mesh_access_collection.second.push_back(identifier);

                auto max_data =
                    model
                        ->accessors
                            [model->meshes[mesh_idx].primitives[primitive_idx].attributes.find("POSITION")->second]
                        .maxValues;
                auto min_data =
                    model
                        ->accessors
                            [model->meshes[mesh_idx].primitives[primitive_idx].attributes.find("POSITION")->second]
                        .minValues;

                bbox[0] = std::min(bbox[0], static_cast<float>(min_data[0]));
                bbox[1] = std::min(bbox[1], static_cast<float>(min_data[1]));
                bbox[2] = std::min(bbox[2], static_cast<float>(min_data[2]));
                bbox[3] = std::max(bbox[3], static_cast<float>(max_data[0]));
                bbox[4] = std::max(bbox[4], static_cast<float>(max_data[1]));
                bbox[5] = std::max(bbox[5], static_cast<float>(max_data[2]));
            }
        }

        if (bbox[0] == std::numeric_limits<float>::max() || bbox[1] == std::numeric_limits<float>::max() ||
            bbox[2] == std::numeric_limits<float>::max() || bbox[3] == std::numeric_limits<float>::min() ||
            bbox[4] == std::numeric_limits<float>::min() || bbox[5] == std::numeric_limits<float>::min()) {

            bbox[0] = -0.5;
            bbox[1] = -0.5;
            bbox[2] = -0.5;
            bbox[3] = 0.5;
            bbox[4] = 0.5;
            bbox[5] = 0.5;
        }

        auto meta_data = lhs_mesh_call->getMetaData();
        meta_data.m_bboxs.SetBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        meta_data.m_bboxs.SetClipBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        lhs_mesh_call->setMetaData(meta_data);
    }

    return true;
}

bool megamol::mesh::GlTFFileLoader::getMeshMetaDataCallback(core::Call& caller) {
    return AbstractMeshDataSource::getMeshMetaDataCallback(caller);
}

bool megamol::mesh::GlTFFileLoader::disconnectGltfCallback(core::Call& caller) {
    return true;
}

bool megamol::mesh::GlTFFileLoader::checkAndLoadGltfModel() {

    if (this->m_glTFFilename_slot.IsDirty()) {
        m_glTFFilename_slot.ResetDirty();

        auto filename = m_glTFFilename_slot.Param<core::param::FilePathParam>()->Value().generic_u8string();
        m_gltf_model = std::make_shared<tinygltf::Model>();

        if (filename != "") {
            tinygltf::TinyGLTF loader;
            std::string err;
            std::string war;

            bool ret = loader.LoadASCIIFromFile(&*m_gltf_model, &err, &war, filename);
            if (!err.empty()) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("Err: %s\n", err.c_str());
            }

            if (!ret) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("Failed to parse glTF\n");
            }
        }

        return true;
    }

    return false;
}

void megamol::mesh::GlTFFileLoader::release() {
    // intentionally empty ?
}
