#include "MSMRenderTaskDataSource.h"

#include <array>

#include "mesh/GPUMeshCollection.h"
#include "mesh/MeshCalls.h"

#include "MSMDataCall.h"

megamol::archvis::MSMRenderTaskDataSource::MSMRenderTaskDataSource()
    : m_MSM_callerSlot("getMSM", "Connects the "), m_MSM_hash(0) {
    this->m_MSM_callerSlot.SetCompatibleCall<MSMDataCallDescription>();
    this->MakeSlotAvailable(&this->m_MSM_callerSlot);
}

megamol::archvis::MSMRenderTaskDataSource::~MSMRenderTaskDataSource() {}

bool megamol::archvis::MSMRenderTaskDataSource::getDataCallback(core::Call& caller) {

    mesh::CallGPURenderTaskData* rtc = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    if (rtc == NULL) return false;

    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
    if (mtlc == NULL) return false;

    if (!(*mtlc)(0)) return false;

    mesh::CallGPUMeshData* mc = this->m_mesh_slot.CallAs<mesh::CallGPUMeshData>();
    if (mc == NULL) return false;

    if (!(*mc)(0)) return false;

    auto gpu_mtl_storage = mtlc->getData();
    auto gpu_mesh_storage = mc->getData();

    if (gpu_mtl_storage == nullptr) return false;
    if (gpu_mesh_storage == nullptr) return false;

    MSMDataCall* msm_call = this->m_MSM_callerSlot.CallAs<MSMDataCall>();
    if (msm_call == NULL) return false;

    if (!(*msm_call)(0)) return false;

    if (this->m_MSM_hash == msm_call->DataHash()) {
        return true;
    }

    m_gpu_render_tasks->clear();


    struct MeshShaderParams {
        vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> transform;
        float force;
        float padding0;
        float padding1;
        float padding2;
    };


    auto msm = msm_call->getMSM();
    auto elem_cnt = msm->getElementCount();

    for (int i = 0; i < elem_cnt; ++i) {
        auto elem_tpye = msm->getElementType(i);

        if (elem_tpye == ScaleModel::STRUT) {
            auto const& gpu_sub_mesh = gpu_mesh_storage->getSubMeshData()[0];
            auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[gpu_sub_mesh.batch_index].mesh;
            auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

            std::vector<glowl::DrawElementsCommand> draw_commands(1, gpu_sub_mesh.sub_mesh_draw_command);
            std::array<MeshShaderParams, 1> obj_data;

            obj_data[0].transform = msm->getElementTransform(i);
            obj_data[0].force = msm->getElementForce(i);

            m_gpu_render_tasks->addRenderTasks(shader, gpu_batch_mesh, draw_commands, obj_data);
        }
        else if (elem_tpye == ScaleModel::DIAGONAL)
        {
            auto const& gpu_sub_mesh = gpu_mesh_storage->getSubMeshData()[1];
            auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[gpu_sub_mesh.batch_index].mesh;
            auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

            std::vector<glowl::DrawElementsCommand> draw_commands(1, gpu_sub_mesh.sub_mesh_draw_command);
            std::array<MeshShaderParams, 1> obj_data;

            obj_data[0].transform = msm->getElementTransform(i);
            obj_data[0].force = msm->getElementForce(i);

            m_gpu_render_tasks->addRenderTasks(shader, gpu_batch_mesh, draw_commands, obj_data);
        } 
        else if (elem_tpye == ScaleModel::FLOOR) 
        {
            auto const& gpu_sub_mesh = gpu_mesh_storage->getSubMeshData()[2];
            auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[gpu_sub_mesh.batch_index].mesh;
            auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

            std::vector<glowl::DrawElementsCommand> draw_commands(1, gpu_sub_mesh.sub_mesh_draw_command);
            std::array<MeshShaderParams, 1> obj_data;

            obj_data[0].transform = msm->getElementTransform(i);
            obj_data[0].force = msm->getElementForce(i);

            m_gpu_render_tasks->addRenderTasks(shader, gpu_batch_mesh, draw_commands, obj_data);
        }
    }


    // for (auto& sub_mesh : gpu_mesh_storage->getSubMeshData()) {
    //    auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;
    //    auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;
    //
    //    std::vector<glowl::DrawElementsCommand> draw_commands(1, sub_mesh.sub_mesh_draw_command);
    //
    //    std::vector<vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR>> object_transform(1000);
    //    typedef std::vector<vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR>> PerTaskData;
    //
    //    GLfloat scale = 1.0f;
    //    object_transform[0].SetAt(0, 0, scale);
    //    object_transform[0].SetAt(1, 1, scale);
    //    object_transform[0].SetAt(2, 2, scale);
    //
    //    object_transform[0].SetAt(3, 3, 1.0f);
    //
    //    object_transform[9].SetAt(0, 3, 0.0f);
    //    object_transform[9].SetAt(1, 3, 0.0f);
    //    object_transform[9].SetAt(2, 3, 0.0f);
    //
    //    m_gpu_render_tasks->addRenderTasks(shader, gpu_batch_mesh, draw_commands, object_transform);
    //}

    rtc->setData(m_gpu_render_tasks);

    this->m_MSM_hash = msm_call->DataHash();

    return true;
}

bool megamol::archvis::MSMRenderTaskDataSource::getMetaDataCallback(core::Call& caller) 
{ 
    megamol::mesh::CallGPURenderTaskData* lhs_rtc = dynamic_cast<megamol::mesh::CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL) return false;

    auto meta_data = lhs_rtc->getMetaData();

    megamol::core::BoundingBoxes bboxs;
    bboxs.SetObjectSpaceBBox(-5.f,-5.0f,-5.0f,5.0f,5.0,5.0f);
    bboxs.SetObjectSpaceClipBox(-5.f, -5.0f, -5.0f, 5.0f, 5.0, 5.0f);

    meta_data.m_frame_cnt = 1;
    meta_data.m_bboxs = bboxs;

    lhs_rtc->setMetaData(meta_data);

    return true; 
}
