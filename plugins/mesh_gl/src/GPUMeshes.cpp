#include "GPUMeshes.h"

#include "mesh_gl/MeshCalls_gl.h"

megamol::mesh_gl::GPUMeshes::GPUMeshes()
        : m_version(0)
        , m_mesh_slot("meshes", "Connect mesh data for upload to the GPU") {
    this->m_mesh_slot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->m_mesh_slot);
}

megamol::mesh_gl::GPUMeshes::~GPUMeshes() {
    this->Release();
}

bool megamol::mesh_gl::GPUMeshes::getDataCallback(core::Call& caller) {
    CallGPUMeshData* lhs_mesh_call = dynamic_cast<CallGPUMeshData*>(&caller);
    CallGPUMeshData* rhs_mesh_call = this->m_mesh_rhs_slot.CallAs<CallGPUMeshData>();

    if (lhs_mesh_call == nullptr) {
        return false;
    }

    std::vector<std::shared_ptr<GPUMeshCollection>> gpu_mesh_collection;
    // if there is a mesh connection to the right, pass on the mesh collection
    if (rhs_mesh_call != nullptr) {
        if (!(*rhs_mesh_call)(0)) {
            return false;
        }
        if (rhs_mesh_call->hasUpdate()) {
            ++m_version;
        }
        gpu_mesh_collection = rhs_mesh_call->getData();
    }
    gpu_mesh_collection.push_back(m_mesh_collection.first);

    mesh::CallMesh* mc = this->m_mesh_slot.CallAs<mesh::CallMesh>();
    if (mc != nullptr) {

        if (!(*mc)(0))
            return false;

        bool something_has_changed = mc->hasUpdate(); // something has changed in the neath...

        if (something_has_changed) {
            ++m_version;

            clearMeshCollection();

            auto meshes = mc->getData()->accessMeshes();

            for (auto& mesh : meshes) {

                // check if primtives type
                GLenum primitive_type = GL_NONE;
                switch (mesh.second.primitive_type) {
                case 0:
                    primitive_type = GL_TRIANGLES;
                    break;
                case 1:
                    primitive_type = GL_PATCHES;
                    break;
                case 2:
                    primitive_type = GL_LINES;
                    break;
                case 3:
                    primitive_type = GL_LINE_STRIP;
                    break;
                case 4:
                    primitive_type = GL_TRIANGLE_FAN;
                    break;
                default:
                    core::utility::log::Log::DefaultLog.WriteError("There was no matching primitive type found!");
                    return false;
                }

                // check if primtives type
                // GLenum primtive_type = GL_TRIANGLES;
                // if (mesh.second.primitive_type == MeshDataAccessCollection::QUADS) {
                //     primtive_type = GL_PATCHES;
                // }

                std::vector<glowl::VertexLayout> vb_layouts;
                std::vector<std::pair<uint8_t*, uint8_t*>> vb_iterators;
                std::pair<uint8_t*, uint8_t*> ib_iterators;

                ib_iterators = {mesh.second.indices.data, mesh.second.indices.data + mesh.second.indices.byte_size};

                auto formated_attrib_indices = mc->getData()->getFormattedAttributeIndices(mesh.first);

                for (auto vb_attribs : formated_attrib_indices) {

                    // for each set of attribute indices, create a vertex buffer layout and set data pointers
                    // using the first attribute (could be any from the set, data pointer and stride should be equal)
                    auto first_attrib = mesh.second.attributes[vb_attribs.front()];
                    vb_layouts.push_back(glowl::VertexLayout(first_attrib.stride, {}));
                    vb_iterators.push_back({first_attrib.data, first_attrib.data + first_attrib.byte_size});

                    // for each attribute in the set, add it to the attributes of the vertex buffer layout
                    for (auto const& attrib_idx : vb_attribs) {
                        auto const& attrib = mesh.second.attributes[attrib_idx];
                        vb_layouts.back().attributes.push_back(glowl::VertexLayout::Attribute(attrib.component_cnt,
                            mesh::MeshDataAccessCollection::convertToGLType(attrib.component_type), GL_FALSE /*ToDO*/,
                            attrib.offset));
                    }
                }

                try {
                    bool store_separate = false;
                    if (mesh.first == "ghostplane")
                        store_separate = true;

                    m_mesh_collection.first->addMesh(mesh.first, vb_layouts, vb_iterators, ib_iterators,
                        mesh::MeshDataAccessCollection::convertToGLType(mesh.second.indices.type), GL_STATIC_DRAW,
                        primitive_type, true);
                    m_mesh_collection.second.push_back(mesh.first);
                } catch (glowl::MeshException const& exc) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Failed to add GPU mesh \"%s\": %s. [%s, %s, line %d]\n", mesh.first.c_str(), exc.what(),
                        __FILE__, __FUNCTION__, __LINE__);

                } catch (glowl::BufferObjectException const& exc) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Failed to add GPU mesh \"%s\": %s. [%s, %s, line %d]\n", mesh.first.c_str(), exc.what(),
                        __FILE__, __FUNCTION__, __LINE__);
                }
            }
        }

        auto lhs_meta_data = lhs_mesh_call->getMetaData();
        core::Spatial3DMetaData rhs_meta_data;
        auto src_meta_data = mc->getMetaData();

        if (rhs_mesh_call != nullptr) {
            rhs_meta_data = rhs_mesh_call->getMetaData();
        } else {
            rhs_meta_data.m_frame_cnt = src_meta_data.m_frame_cnt;
        }

        lhs_meta_data.m_frame_cnt = std::min(src_meta_data.m_frame_cnt, rhs_meta_data.m_frame_cnt);

        auto bbox = src_meta_data.m_bboxs.BoundingBox();
        bbox.Union(rhs_meta_data.m_bboxs.BoundingBox());
        lhs_meta_data.m_bboxs.SetBoundingBox(bbox);

        auto cbbox = src_meta_data.m_bboxs.ClipBox();
        cbbox.Union(rhs_meta_data.m_bboxs.ClipBox());
        lhs_meta_data.m_bboxs.SetClipBox(cbbox);

        lhs_mesh_call->setMetaData(lhs_meta_data);
    } else {
        clearMeshCollection();

        ++m_version;
    }

    if (lhs_mesh_call->version() < m_version) {
        lhs_mesh_call->setData(gpu_mesh_collection, m_version);
    }

    return true;
}

bool megamol::mesh_gl::GPUMeshes::getMetaDataCallback(core::Call& caller) {
    CallGPUMeshData* lhs_mesh_call = dynamic_cast<CallGPUMeshData*>(&caller);
    CallGPUMeshData* rhs_mesh_call = m_mesh_rhs_slot.CallAs<CallGPUMeshData>();
    mesh::CallMesh* src_mesh_call = m_mesh_slot.CallAs<mesh::CallMesh>();

    if (lhs_mesh_call == NULL)
        return false;
    if (src_mesh_call == NULL)
        return false;

    auto lhs_meta_data = lhs_mesh_call->getMetaData();
    auto src_meta_data = src_mesh_call->getMetaData();
    core::Spatial3DMetaData rhs_meta_data;

    src_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
    src_mesh_call->setMetaData(src_meta_data);
    if (!(*src_mesh_call)(1))
        return false;
    src_meta_data = src_mesh_call->getMetaData();

    if (rhs_mesh_call != NULL) {
        rhs_meta_data = rhs_mesh_call->getMetaData();
        rhs_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
        rhs_mesh_call->setMetaData(rhs_meta_data);
        if (!(*rhs_mesh_call)(1))
            return false;
        rhs_meta_data = rhs_mesh_call->getMetaData();
    } else {
        rhs_meta_data.m_frame_cnt = 1;
    }

    lhs_meta_data.m_frame_cnt = std::min(src_meta_data.m_frame_cnt, rhs_meta_data.m_frame_cnt);

    auto bbox = src_meta_data.m_bboxs.BoundingBox();
    bbox.Union(rhs_meta_data.m_bboxs.BoundingBox());
    lhs_meta_data.m_bboxs.SetBoundingBox(bbox);

    auto cbbox = src_meta_data.m_bboxs.ClipBox();
    cbbox.Union(rhs_meta_data.m_bboxs.ClipBox());
    lhs_meta_data.m_bboxs.SetClipBox(cbbox);

    lhs_mesh_call->setMetaData(lhs_meta_data);

    return true;
}
