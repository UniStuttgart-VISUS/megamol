#include "FEMRenderTaskDataSource.h"

#include <variant>

#include "mesh_gl/GPUMeshCollection.h"
#include "mesh_gl/MeshCalls_gl.h"

#include "ArchVisCalls.h"

megamol::archvis::FEMRenderTaskDataSource::FEMRenderTaskDataSource()
        : m_fem_callerSlot("getFEMFile", "Connects the data source with loaded FEM data")
        , m_version(0) {
    this->m_fem_callerSlot.SetCompatibleCall<FEMModelCallDescription>();
    this->MakeSlotAvailable(&this->m_fem_callerSlot);
}

megamol::archvis::FEMRenderTaskDataSource::~FEMRenderTaskDataSource() {}

bool megamol::archvis::FEMRenderTaskDataSource::create(void) {
    mesh_gl::AbstractGPURenderTaskDataSource::create();

    m_material_collection = std::make_shared<mesh_gl::GPUMaterialCollection>();
    std::vector<std::filesystem::path> shaderfiles = {"archvis/FEM/fem_vert.glsl", "archvis/FEM/fem_geom.glsl", "archvis/FEM/fem_frag.glsl"};
    m_material_collection->addMaterial(this->instance(), "ArchVisFEM", shaderfiles);

    return true;
}

bool megamol::archvis::FEMRenderTaskDataSource::getDataCallback(core::Call& caller) {
    mesh_gl::CallGPURenderTaskData* lhs_rtc = dynamic_cast<mesh_gl::CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == nullptr) {
        return false;
    }

    mesh_gl::CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<mesh_gl::CallGPURenderTaskData>();

    auto gpu_render_tasks = std::make_shared<std::vector<std::shared_ptr<mesh_gl::GPURenderTaskCollection>>>();
    if (rhs_rtc != nullptr) {
        if (!(*rhs_rtc)(0)) {
            return false;
        }
        if (rhs_rtc->hasUpdate()) {
            ++m_version;
        }
        gpu_render_tasks = rhs_rtc->getData();
    }
    gpu_render_tasks->push_back(m_rendertask_collection.first);


    mesh_gl::CallGPUMeshData* mc = this->m_mesh_slot.CallAs<mesh_gl::CallGPUMeshData>();
    if (mc == nullptr) {
        return false;
    }
    if (!(*mc)(0)) {
        return false;
    }

    CallFEMModel* fem_call = this->m_fem_callerSlot.CallAs<CallFEMModel>();
    if (fem_call == nullptr) {
        return false;
    }
    if (!(*fem_call)(0)) {
        return false;
    }

    if (mc->hasUpdate() || fem_call->hasUpdate()) {
        ++m_version;

        clearRenderTaskCollection();

        auto gpu_mesh_storage = mc->getData();


        for (auto& gpu_mesh_collection : *gpu_mesh_storage) {
            for (auto& sub_mesh : gpu_mesh_collection->getSubMeshData()) {
                auto const& gpu_batch_mesh = sub_mesh.second.mesh->mesh;

                auto const& shader = m_material_collection->getMaterial("ArchVisFEM").shader_program;

                vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transform;
                typedef std::vector<vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR>> PerTaskData;

                GLfloat scale = 1.0f;
                object_transform.SetAt(0, 0, scale);
                object_transform.SetAt(1, 1, scale);
                object_transform.SetAt(2, 2, scale);

                object_transform.SetAt(3, 3, 1.0f);

                object_transform.SetAt(0, 3, 0.0f);
                object_transform.SetAt(1, 3, 0.0f);
                object_transform.SetAt(2, 3, 0.0f);

                auto identifier = std::string(this->FullName()) + sub_mesh.first;
                m_rendertask_collection.first->addRenderTask(
                    identifier, shader, gpu_batch_mesh, sub_mesh.second.sub_mesh_draw_command, object_transform);
                m_rendertask_collection.second.push_back(identifier);
            }
        }


        auto const& node_deformation = fem_call->getData()->getDynamicData();

        m_rendertask_collection.first->addPerFrameDataBuffer("", node_deformation, 1);

        //{
        //    // TODO get transfer function texture and add as per frame data
        //    std::vector<GLuint64> texture_handles;
        //    auto textures = gpu_mtl_storage->getMaterials().front().textures;
        //    for (auto texture : textures) {
        //
        //        texture_handles.push_back(texture->getTextureHandle());
        //        // base_texture->makeResident();
        //    }
        //    m_gpu_render_tasks->updatePerFrameDataBuffer("", texture_handles, 2);
        //}
    }

    lhs_rtc->setData(gpu_render_tasks, m_version);

    return true;
}
