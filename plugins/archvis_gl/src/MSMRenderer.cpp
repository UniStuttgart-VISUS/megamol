/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "MSMRenderer.h"

#include <array>

#include "ArchVisCalls.h"
#include "MSMRenderer.h"
#include "mesh_gl/GPUMeshCollection.h"

megamol::archvis_gl::MSMRenderer::MSMRenderer() : m_MSM_callerSlot("getMSM", "Connects...") {
    this->m_MSM_callerSlot.SetCompatibleCall<ScaleModelCallDescription>();
    this->MakeSlotAvailable(&this->m_MSM_callerSlot);
}

megamol::archvis_gl::MSMRenderer::~MSMRenderer() {}

bool megamol::archvis_gl::MSMRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
    megamol::core::BoundingBoxes bboxs;
    bboxs.SetObjectSpaceBBox(-5.f, -5.0f, -5.0f, 5.0f, 5.0, 5.0f);
    bboxs.SetObjectSpaceClipBox(-5.f, -5.0f, -5.0f, 5.0f, 5.0, 5.0f);

    call.SetTimeFramesCount(1);
    call.AccessBoundingBoxes() = bboxs;

    return true;
}

void megamol::archvis_gl::MSMRenderer::createMaterialCollection() {
    material_collection_ = std::make_shared<mesh_gl::GPUMaterialCollection>();
    material_collection_->addMaterial(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>(),
        "scalemodel", {"archvis_gl/scale_model.vert.glsl", "archvis_gl/scale_model.frag.glsl"});
}

void megamol::archvis_gl::MSMRenderer::updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) {

    bool something_has_changed = force_update;

    CallScaleModel* msm_call = this->m_MSM_callerSlot.CallAs<CallScaleModel>();
    if (msm_call != nullptr) {

        if (!(*msm_call)(0)) {
            return;
        }

        something_has_changed |= msm_call->hasUpdate();

        if (something_has_changed) {
            render_task_collection_->clear();

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

                    sub_mesh = mesh_collection_->getSubMesh("strut");

                    if (sub_mesh.mesh != nullptr) {
                        break;
                    }
                } else if (elem_tpye == ScaleModel::DIAGONAL) {

                    sub_mesh = mesh_collection_->getSubMesh("diagonal");

                    if (sub_mesh.mesh != nullptr) {
                        break;
                    }
                } else if (elem_tpye == ScaleModel::FLOOR) {

                    sub_mesh = mesh_collection_->getSubMesh("floor");

                    if (sub_mesh.mesh != nullptr) {
                        break;
                    }
                }

                if (sub_mesh.mesh != nullptr) {
                    // TODO throw error
                }

                auto const& gpu_batch_mesh = sub_mesh.mesh->mesh;
                auto const& shader = material_collection_->getMaterial("scalemodel").shader_program;

                MeshShaderParams obj_data;

                obj_data.transform = msm->getElementTransform(i);
                obj_data.force = msm->getElementForce(i);

                std::string rt_identifier(std::string(this->FullName()) + "_" + std::to_string(i));
                render_task_collection_->addRenderTask(
                    rt_identifier, shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, obj_data);
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
    }
}
