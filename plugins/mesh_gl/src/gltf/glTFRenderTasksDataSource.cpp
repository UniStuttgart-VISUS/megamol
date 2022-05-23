
#include "glTFRenderTasksDataSource.h"
#include "mmcore/param/StringParam.h"
#include "tiny_gltf.h"
#include "vislib/math/Matrix.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"

#include "mesh_gl/MeshCalls_gl.h"

megamol::mesh_gl::GlTFRenderTasksDataSource::GlTFRenderTasksDataSource()
        : m_version(0)
        , m_glTF_callerSlot("gltfModels", "Connects a collection of loaded glTF files")
        , m_material_collection(nullptr)
        , m_btf_name_slot("BTF name", "Overload default gltf shader") {
    this->m_glTF_callerSlot.SetCompatibleCall<mesh::CallGlTFDataDescription>();
    this->MakeSlotAvailable(&this->m_glTF_callerSlot);

    this->m_btf_name_slot << new core::param::StringParam("dfr_gltfExample");
    this->MakeSlotAvailable(&this->m_btf_name_slot);
}

megamol::mesh_gl::GlTFRenderTasksDataSource::~GlTFRenderTasksDataSource() {}

bool megamol::mesh_gl::GlTFRenderTasksDataSource::create(void) {
    AbstractGPURenderTaskDataSource::create();

    m_material_collection = std::make_shared<GPUMaterialCollection>();
    m_material_collection->addMaterial(this->instance(), "dfr_gltfExample", "dfr_gltfExample");

    return true;
}

bool megamol::mesh_gl::GlTFRenderTasksDataSource::getDataCallback(core::Call& caller) {
    CallGPURenderTaskData* lhs_rtc = dynamic_cast<CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == nullptr) {
        return false;
    }

    CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<CallGPURenderTaskData>();

    std::vector<std::shared_ptr<GPURenderTaskCollection>> gpu_render_tasks;
    if (rhs_rtc != nullptr) {
        if (!(*rhs_rtc)(0)) {
            return false;
        }
        if (rhs_rtc->hasUpdate()) {
            ++m_version;
        }
        gpu_render_tasks = rhs_rtc->getData();
    }
    gpu_render_tasks.push_back(m_rendertask_collection.first);

    CallGPUMeshData* mc = this->m_mesh_slot.CallAs<CallGPUMeshData>();
    mesh::CallGlTFData* gltf_call = this->m_glTF_callerSlot.CallAs<mesh::CallGlTFData>();

    if (mc != nullptr && gltf_call != nullptr) {
        if (!(*mc)(0)) {
            return false;
        }

        if (!(*gltf_call)(0)) {
            return false;
        }

        auto gpu_mesh_storage = mc->getData();

        if (gltf_call->hasUpdate() || this->m_btf_name_slot.IsDirty()) {
            ++m_version;

            if (this->m_btf_name_slot.IsDirty()) {
                m_btf_name_slot.ResetDirty();
                auto filename = m_btf_name_slot.Param<core::param::StringParam>()->Value();
                m_material_collection->clear();
                m_material_collection->addMaterial(this->instance(), filename, filename);
            }

            clearRenderTaskCollection();

            auto model = gltf_call->getData().second;

            for (size_t node_idx = 0; node_idx < model->nodes.size(); node_idx++) {
                if (node_idx < model->nodes.size() && model->nodes[node_idx].mesh != -1) {
                    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transform;

                    if (model->nodes[node_idx].matrix.size() != 0) // has matrix transform
                    {
                        // TODO
                    } else {
                        auto& translation = model->nodes[node_idx].translation;
                        auto& scale = model->nodes[node_idx].scale;
                        auto& rotation = model->nodes[node_idx].rotation;

                        if (translation.size() != 0) {
                            object_transform.SetAt(0, 3, translation[0]);
                            object_transform.SetAt(1, 3, translation[1]);
                            object_transform.SetAt(2, 3, translation[2]);
                        }

                        if (scale.size() != 0) {}

                        if (rotation.size() != 0) {}
                    }

                    auto primitive_cnt = model->meshes[model->nodes[node_idx].mesh].primitives.size();
                    for (size_t primitive_idx = 0; primitive_idx < primitive_cnt; ++primitive_idx) {
                        std::string sub_mesh_identifier = gltf_call->getData().first +
                                                          model->meshes[model->nodes[node_idx].mesh].name + "_" +
                                                          std::to_string(primitive_idx);

                        GPUMeshCollection::SubMeshData sub_mesh;
                        for (auto const& gpu_mesh_collection : gpu_mesh_storage) {
                            sub_mesh = gpu_mesh_collection->getSubMesh(sub_mesh_identifier);

                            if (sub_mesh.mesh != nullptr) {
                                break;
                            }
                        }

                        //TODO will use its own material storage in the future
                        if (sub_mesh.mesh != nullptr && !m_material_collection->getMaterials().empty()) {
                            auto const& gpu_batch_mesh = sub_mesh.mesh->mesh;
                            auto const& shader = m_material_collection->getMaterials().begin()->second.shader_program;

                            std::string rt_identifier(std::string(this->FullName()) + "_" + sub_mesh_identifier);
                            m_rendertask_collection.first->addRenderTask(rt_identifier, shader, gpu_batch_mesh,
                                sub_mesh.sub_mesh_draw_command, object_transform);
                            m_rendertask_collection.second.push_back(rt_identifier);
                        }
                    }
                }
            }
        }
    } else {
        clearRenderTaskCollection();
    }

    lhs_rtc->setData(gpu_render_tasks, m_version);

    return true;
}

bool megamol::mesh_gl::GlTFRenderTasksDataSource::getMetaDataCallback(core::Call& caller) {

    AbstractGPURenderTaskDataSource::getMetaDataCallback(caller);

    auto gltf_call = m_glTF_callerSlot.CallAs<mesh::CallGlTFData>();

    if (gltf_call == nullptr) {
        return false;
    }

    if (!(*gltf_call)(1)) {
        return false;
    }

    return true;
}
