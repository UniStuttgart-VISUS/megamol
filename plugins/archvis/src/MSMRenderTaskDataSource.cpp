#include "MSMRenderTaskDataSource.h"

#include "mesh/CallGPUMaterialData.h"
#include "mesh/GPUMeshCollection.h"
#include "mesh/CallGPUMeshData.h"
#include "mesh/CallGPURenderTaskData.h"

#include "MSMDataCall.h"

megamol::archvis::MSMRenderTaskDataSource::MSMRenderTaskDataSource()
    : m_MSM_callerSlot("getMSM", "Connects the "), m_MSM_hash(0) {
    this->m_MSM_callerSlot.SetCompatibleCall<MSMDataCallDescription>();
    this->MakeSlotAvailable(&this->m_MSM_callerSlot);
}

megamol::archvis::MSMRenderTaskDataSource::~MSMRenderTaskDataSource(){
}

bool megamol::archvis::MSMRenderTaskDataSource::getDataCallback(core::Call& caller) {

    mesh::CallGPURenderTaskData* rtc = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    if (rtc == NULL) return false;

    mesh::CallGPUMaterialData* mtlc = this->m_material_callerSlot.CallAs<mesh::CallGPUMaterialData>();
    if (mtlc == NULL) return false;

    if (!(*mtlc)(0)) return false;

    mesh::CallGPUMeshData* mc = this->m_mesh_callerSlot.CallAs<mesh::CallGPUMeshData>();
    if (mc == NULL) return false;

    if (!(*mc)(0)) return false;

    auto gpu_mtl_storage = mtlc->getMaterialStorage();
    auto gpu_mesh_storage = mc->getGPUMeshes();

    if (gpu_mtl_storage == nullptr) return false;
    if (gpu_mesh_storage == nullptr) return false;

    MSMDataCall* msm_call = this->m_MSM_callerSlot.CallAs<MSMDataCall>();
    if (msm_call == NULL) return false;

    if (!(*msm_call)(0)) return false;


    if (this->m_MSM_hash == msm_call->DataHash()) {
        return true;
    }

    m_gpu_render_tasks->clear();

    for (auto& sub_mesh : gpu_mesh_storage->getSubMeshData()) {
        auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;
        auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

        std::vector<mesh::DrawElementsCommand> draw_commands(1, sub_mesh.sub_mesh_draw_command);

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

    rtc->setRenderTaskData(m_gpu_render_tasks);

    this->m_MSM_hash = msm_call->DataHash();

    return true;
}
