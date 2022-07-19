/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "MSMRenderTaskDataSource.h"

#include <array>

#include "ArchVisCalls.h"
#include "mesh_gl/GPUMeshCollection.h"
#include "mesh_gl/MeshCalls_gl.h"

megamol::archvis_gl::MSMRenderTaskDataSource::MSMRenderTaskDataSource()
        : m_MSM_callerSlot("getMSM", "Connects the ")
        , m_version(0) {
    this->m_MSM_callerSlot.SetCompatibleCall<ScaleModelCallDescription>();
    this->MakeSlotAvailable(&this->m_MSM_callerSlot);
}

megamol::archvis_gl::MSMRenderTaskDataSource::~MSMRenderTaskDataSource() {}

bool megamol::archvis_gl::MSMRenderTaskDataSource::getDataCallback(core::Call& caller) {
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

    CallScaleModel* msm_call = this->m_MSM_callerSlot.CallAs<CallScaleModel>();
    if (msm_call == nullptr) {
        return false;
    }

    if (!(*mc)(0)) {
        return false;
    }
    if (!(*msm_call)(0)) {
        return false;
    }

    if (mc->hasUpdate() || msm_call->hasUpdate()) {
        ++m_version;

        clearRenderTaskCollection();

        auto gpu_mesh_storage = mc->getData();

        struct MeshShaderParams {
            vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> transform;
            float force;
            float padding0;
            float padding1;
            float padding2;
        };

        auto msm = msm_call->getData();
        auto elem_cnt = msm->getElementCount();

        for (int i = 0; i < elem_cnt; ++i) {
            auto elem_tpye = msm->getElementType(i);

            mesh_gl::GPUMeshCollection::SubMeshData sub_mesh;
            if (elem_tpye == ScaleModel::STRUT) {
                for (auto const& gpu_mesh_collection : *gpu_mesh_storage) {
                    sub_mesh = gpu_mesh_collection->getSubMesh("strut");

                    if (sub_mesh.mesh != nullptr) {
                        break;
                    }
                }
            } else if (elem_tpye == ScaleModel::DIAGONAL) {
                for (auto const& gpu_mesh_collection : *gpu_mesh_storage) {
                    sub_mesh = gpu_mesh_collection->getSubMesh("diagonal");

                    if (sub_mesh.mesh != nullptr) {
                        break;
                    }
                }
            } else if (elem_tpye == ScaleModel::FLOOR) {
                for (auto const& gpu_mesh_collection : *gpu_mesh_storage) {
                    sub_mesh = gpu_mesh_collection->getSubMesh("floor");

                    if (sub_mesh.mesh != nullptr) {
                        break;
                    }
                }
            }

            auto const& gpu_batch_mesh = sub_mesh.mesh->mesh;
            auto const& shader = m_material_collection->getMaterial("scalemodel").shader_program;

            MeshShaderParams obj_data;

            obj_data.transform = msm->getElementTransform(i);
            obj_data.force = msm->getElementForce(i);

            std::string rt_identifier(std::string(this->FullName()) + "_" + std::to_string(i));
            m_rendertask_collection.first->addRenderTask(
                rt_identifier, shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, obj_data);
            m_rendertask_collection.second.push_back(rt_identifier);
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
    }

    lhs_rtc->setData(gpu_render_tasks, m_version);

    return true;
}

bool megamol::archvis_gl::MSMRenderTaskDataSource::getMetaDataCallback(core::Call& caller) {
    megamol::mesh_gl::CallGPURenderTaskData* lhs_rtc = dynamic_cast<megamol::mesh_gl::CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL)
        return false;

    auto meta_data = lhs_rtc->getMetaData();

    megamol::core::BoundingBoxes bboxs;
    bboxs.SetObjectSpaceBBox(-5.f, -5.0f, -5.0f, 5.0f, 5.0, 5.0f);
    bboxs.SetObjectSpaceClipBox(-5.f, -5.0f, -5.0f, 5.0f, 5.0, 5.0f);

    meta_data.m_frame_cnt = 1;
    meta_data.m_bboxs = bboxs;

    lhs_rtc->setMetaData(meta_data);

    return true;
}
