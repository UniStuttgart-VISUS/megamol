#include "stdafx.h"

#include "glTFMeshesDataSource.h"
#include "mesh/CallGPUMeshData.h"
#include "mesh/CallGltfData.h"
#include "tiny_gltf.h"

megamol::mesh::GlTFMeshesDataSource::GlTFMeshesDataSource()
    : m_glTF_callerSlot("getGlTFFile", "Connects the data source with a loaded glTF file") {
    this->m_glTF_callerSlot.SetCompatibleCall<CallGlTFDataDescription>();
    this->MakeSlotAvailable(&this->m_glTF_callerSlot);
}

megamol::mesh::GlTFMeshesDataSource::~GlTFMeshesDataSource() {}

bool megamol::mesh::GlTFMeshesDataSource::create() {
    m_gpu_meshes = std::make_shared<GPUMeshCollection>();

    m_bbox = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

    return true;
}

bool megamol::mesh::GlTFMeshesDataSource::getDataCallback(core::Call& caller) {
    CallGPUMeshData* lhs_mesh_call = dynamic_cast<CallGPUMeshData*>(&caller);
    if (lhs_mesh_call == NULL) return false;

    std::shared_ptr<GPUMeshCollection> mesh_collection(nullptr);

    if (lhs_mesh_call->getGPUMeshes() == nullptr) {
        // no incoming material -> use your own material storage
        mesh_collection = this->m_gpu_meshes;
        lhs_mesh_call->setGPUMeshes(mesh_collection);
    } else {
        // incoming material -> use it (delete local?)
        mesh_collection = lhs_mesh_call->getGPUMeshes();
    }

    CallGlTFData* gltf_call = this->m_glTF_callerSlot.CallAs<CallGlTFData>();
    if (gltf_call == NULL) return false;

    if (!(*gltf_call)(0)) return false;

    if (gltf_call->getUpdateFlag()) {
        m_gpu_meshes->clear();

        m_bbox[0] = std::numeric_limits<float>::max();
        m_bbox[1] = std::numeric_limits<float>::max();
        m_bbox[2] = std::numeric_limits<float>::max();
        m_bbox[3] = std::numeric_limits<float>::min();
        m_bbox[4] = std::numeric_limits<float>::min();
        m_bbox[5] = std::numeric_limits<float>::min();

        auto model = gltf_call->getGlTFModel();

        for (size_t mesh_idx = 0; mesh_idx < model->meshes.size(); mesh_idx++) {
            auto primitive_cnt = model->meshes[mesh_idx].primitives.size();

            for (size_t primitive_idx = 0; primitive_idx < primitive_cnt; ++primitive_idx) {
                std::vector<VertexLayout::Attribute> attribs;
                std::vector<std::pair<std::vector<unsigned char>::iterator, std::vector<unsigned char>::iterator>>
                    vb_iterators;
                std::pair<std::vector<unsigned char>::iterator, std::vector<unsigned char>::iterator> ib_iterators;

                auto& indices_accessor = model->accessors[model->meshes[mesh_idx].primitives[primitive_idx].indices];
                auto& indices_bufferView = model->bufferViews[indices_accessor.bufferView];
                auto& indices_buffer = model->buffers[indices_bufferView.buffer];

                ib_iterators = {
                    indices_buffer.data.begin() + indices_bufferView.byteOffset + indices_accessor.byteOffset,
                    indices_buffer.data.begin() + indices_bufferView.byteOffset + indices_accessor.byteOffset +
                        (indices_accessor.count * indices_accessor.ByteStride(indices_bufferView))};

                auto& vertex_attributes = model->meshes[mesh_idx].primitives[primitive_idx].attributes;
                for (auto attrib : vertex_attributes) {
                    auto& vertexAttrib_accessor = model->accessors[attrib.second];
                    auto& vertexAttrib_bufferView = model->bufferViews[vertexAttrib_accessor.bufferView];
                    auto& vertexAttrib_buffer = model->buffers[vertexAttrib_bufferView.buffer];

                    attribs.push_back(
                        VertexLayout::Attribute(vertexAttrib_accessor.type, vertexAttrib_accessor.componentType,
                            vertexAttrib_accessor.normalized, vertexAttrib_accessor.byteOffset));

                    // TODO vb_iterators
                    vb_iterators.push_back({vertexAttrib_buffer.data.begin() + vertexAttrib_bufferView.byteOffset +
                                                vertexAttrib_accessor.byteOffset,
                        vertexAttrib_buffer.data.begin() + vertexAttrib_bufferView.byteOffset +
                            vertexAttrib_accessor.byteOffset +
                            (vertexAttrib_accessor.count * vertexAttrib_accessor.ByteStride(vertexAttrib_bufferView))});
                }

                VertexLayout vertex_descriptor(0, attribs);
                m_gpu_meshes->addMesh(vertex_descriptor, vb_iterators, ib_iterators, indices_accessor.componentType,
                    GL_STATIC_DRAW, GL_TRIANGLES);

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

                m_bbox[0] = std::min(m_bbox[0], static_cast<float>(min_data[0]));
                m_bbox[1] = std::min(m_bbox[1], static_cast<float>(min_data[1]));
                m_bbox[2] = std::min(m_bbox[2], static_cast<float>(min_data[2]));
                m_bbox[3] = std::max(m_bbox[3], static_cast<float>(max_data[0]));
                m_bbox[4] = std::max(m_bbox[4], static_cast<float>(max_data[1]));
                m_bbox[5] = std::max(m_bbox[5], static_cast<float>(max_data[2]));
            }
        }

        // set update_all_flag?
    }

    // if there is a material connection to the right, pass on the material collection
    CallGPUMeshData* rhs_mesh_call = this->m_mesh_callerSlot.CallAs<CallGPUMeshData>();
    if (rhs_mesh_call != NULL) {
        rhs_mesh_call->setGPUMeshes(mesh_collection);
    }

    return true;
}
