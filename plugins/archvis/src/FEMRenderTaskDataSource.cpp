#include "FEMRenderTaskDataSource.h"

#include <variant>

#include "mesh/GPUMeshCollection.h"
#include "mesh/MeshCalls.h"

#include "ArchVisCalls.h"

megamol::archvis::FEMRenderTaskDataSource::FEMRenderTaskDataSource()
    : m_fem_callerSlot("getFEMFile", "Connects the data source with loaded FEM data"), m_FEM_model_hash(0) {
    this->m_fem_callerSlot.SetCompatibleCall<FEMModelCallDescription>();
    this->MakeSlotAvailable(&this->m_fem_callerSlot);
}

megamol::archvis::FEMRenderTaskDataSource::~FEMRenderTaskDataSource() {}

bool megamol::archvis::FEMRenderTaskDataSource::getDataCallback(core::Call& caller) {
    mesh::CallGPURenderTaskData* lhs_rtc = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == nullptr) {
        return false;
    }

    mesh::CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<mesh::CallGPURenderTaskData>();

    std::vector<std::shared_ptr<mesh::GPURenderTaskCollection>> gpu_render_tasks;
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


    mesh::CallGPUMeshData* mc = this->m_mesh_slot.CallAs<mesh::CallGPUMeshData>();
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

        for (auto& sub_mesh : gpu_mesh_storage->getSubMeshData()) {
            auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;
            auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

            std::vector<glowl::DrawElementsCommand> draw_commands(1, sub_mesh.sub_mesh_draw_command);

            std::vector<vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR>> object_transform(1000);
            typedef std::vector<vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR>> PerTaskData;

            GLfloat scale = 1.0f;
            object_transform[0].SetAt(0, 0, scale);
            object_transform[0].SetAt(1, 1, scale);
            object_transform[0].SetAt(2, 2, scale);

            object_transform[0].SetAt(3, 3, 1.0f);

            object_transform[9].SetAt(0, 3, 0.0f);
            object_transform[9].SetAt(1, 3, 0.0f);
            object_transform[9].SetAt(2, 3, 0.0f);

            m_gpu_render_tasks->addRenderTasks(shader, gpu_batch_mesh, draw_commands, object_transform);
        }

        auto const& node_deformation = fem_call->getData()->getDynamicData();

        m_rendertask_collection.first->addPerFrameDataBuffer("",node_deformation, 1);

        {
            // TODO get transfer function texture and add as per frame data
            std::vector<GLuint64> texture_handles;
            auto textures = gpu_mtl_storage->getMaterials().front().textures;
            for (auto texture : textures) {

                texture_handles.push_back(texture->getTextureHandle());
                // base_texture->makeResident();
            }
            m_gpu_render_tasks->updatePerFrameDataBuffer("", texture_handles, 2);
        }
    }

    lhs_rtc->setData(gpu_render_tasks,m_version);

    return true;
}
