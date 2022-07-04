/*
 * MeshViewerRenderTasks.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "MeshViewerRenderTasks.h"

#include "mesh_gl/MeshCalls_gl.h"

megamol::mesh_gl::MeshViewerRenderTasks::MeshViewerRenderTasks()
        : m_version(0)
        , m_material_slot("gpuMaterials", "Connects to a material data source") {
    this->m_material_slot.SetCompatibleCall<CallGPUMaterialDataDescription>();
    this->MakeSlotAvailable(&this->m_material_slot);
}

megamol::mesh_gl::MeshViewerRenderTasks::~MeshViewerRenderTasks() {}

bool megamol::mesh_gl::MeshViewerRenderTasks::getDataCallback(core::Call& caller) {

    CallGPURenderTaskData* lhs_rtc = dynamic_cast<CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == nullptr) {
        return false;
    }

    CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<CallGPURenderTaskData>();

    auto gpu_render_tasks = std::make_shared<std::vector<std::shared_ptr<GPURenderTaskCollection>>>();
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

    CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<CallGPUMaterialData>();
    CallGPUMeshData* mc = this->m_mesh_slot.CallAs<CallGPUMeshData>();

    if (mtlc != nullptr && mc != nullptr) {
        if (!(*mtlc)(0))
            return false;
        if (!(*mc)(0))
            return false;

        bool something_has_changed = mtlc->hasUpdate() || mc->hasUpdate();

        if (something_has_changed) {
            ++m_version;

            clearRenderTaskCollection();

            auto gpu_mtl_collections = mtlc->getData();
            auto gpu_mesh_collections = mc->getData();

            if (gpu_mtl_collections != nullptr && gpu_mesh_collections != nullptr) {
                std::vector<std::vector<std::string>> identifiers;
                std::vector<std::vector<glowl::DrawElementsCommand>> draw_commands;
                std::vector<std::vector<std::array<float, 16>>> object_transforms;
                std::vector<std::shared_ptr<glowl::Mesh>> batch_meshes;

                std::shared_ptr<glowl::Mesh> prev_mesh(nullptr);

                for (auto& gpu_mesh_collection : (*gpu_mesh_collections)) {
                    for (auto& sub_mesh : gpu_mesh_collection->getSubMeshData()) {
                        auto const& gpu_batch_mesh = sub_mesh.second.mesh->mesh;

                        if (gpu_batch_mesh != prev_mesh) {
                            identifiers.emplace_back(std::vector<std::string>());
                            draw_commands.emplace_back(std::vector<glowl::DrawElementsCommand>());
                            object_transforms.emplace_back(std::vector<std::array<float, 16>>());
                            batch_meshes.push_back(gpu_batch_mesh);

                            prev_mesh = gpu_batch_mesh;
                        }

                        float scale = 1.0f;
                        std::array<float, 16> obj_xform = {scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f,
                            scale, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

                        identifiers.back().emplace_back(std::string(this->FullName()) + sub_mesh.first);
                        draw_commands.back().push_back(sub_mesh.second.sub_mesh_draw_command);
                        object_transforms.back().push_back(obj_xform);
                    }
                }
                // just take the first available shader as shader selection will eventually be integrated into this module
                if (gpu_mtl_collections->size() > 0 && gpu_mtl_collections->front()->getMaterials().size() > 0) {

                }
                for (int i = 0; i < batch_meshes.size(); ++i) {
                    auto const& shader = gpu_mtl_collections->front()->getMaterials().begin()->second.shader_program;
                    m_rendertask_collection.first->addRenderTasks(
                        identifiers[i], shader, batch_meshes[i], draw_commands[i], object_transforms[i]);
                    m_rendertask_collection.second.push_back(std::string(this->FullName()));
                }
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Nullpointer given for mesh or material data to \"%s\" [%s, %s, line %d]\n", this->FullName(), __FILE__,
                    __FUNCTION__, __LINE__);
            }
        }
    } else {
        clearRenderTaskCollection();
    }

    // TODO merge meta data stuff, i.e. bounding box

    // set data if necessary
    lhs_rtc->setData(gpu_render_tasks, m_version);

    return true;
}
