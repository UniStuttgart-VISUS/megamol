/*
 * ProbeHullRenderTasks.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmstd/event/EventCall.h"

#include "ProbeEvents.h"
#include "ProbeGlCalls.h"
#include "ProbeHUllRenderTasks.h"
#include "probe/ProbeCalls.h"

#include "mesh/MeshCalls.h"

bool megamol::probe_gl::ProbeHullRenderTasks::create() {
    auto retval = AbstractGPURenderTaskDataSource::create();

    struct PerFrameData {
        int shading_mode;
    };
    std::array<PerFrameData, 1> per_frame_data;
    per_frame_data[0].shading_mode = m_shading_mode_slot.Param<core::param::EnumParam>()->Value();
    m_rendertask_collection.first->addPerFrameDataBuffer("", per_frame_data, 1);

    m_material_collection = std::make_shared<mesh_gl::GPUMaterialCollection>();
    try {
        std::vector<std::filesystem::path> shaderfiles = {"hull/dfr_hull_patch.vert.glsl", "hull/dfr_hull.frag.glsl",
            "hull/dfr_hull.tesc.glsl", "hull/dfr_hull.tese.glsl"};
        m_material_collection->addMaterial(
            frontend_resources.get<megamol::frontend_resources::RuntimeConfig>(), "ProbeHull", shaderfiles);
    } catch (const std::exception& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "%s [%s, %s, line %d]\n", ex.what(), __FILE__, __FUNCTION__, __LINE__);
        retval = false;
    }
    try {
        std::vector<std::filesystem::path> shaderfiles = {"hull/dfr_hull_tri.vert.glsl", "hull/dfr_hull.frag.glsl"};
        m_material_collection->addMaterial(
            frontend_resources.get<megamol::frontend_resources::RuntimeConfig>(), "ProbeTriangleHull", shaderfiles);
    } catch (const std::exception& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "%s [%s, %s, line %d]\n", ex.what(), __FILE__, __FUNCTION__, __LINE__);
        retval = false;
    }

    return retval;
}

megamol::probe_gl::ProbeHullRenderTasks::ProbeHullRenderTasks()
        : m_version(0)
        , m_show_hull(true)
        //, m_probes_slot("probes","")
        , m_event_slot("GetEvents", "")
        , m_shading_mode_slot("ShadingMode", "")
        , m_hull_color_slot("HullColor", "") {
    //this->m_probes_slot.SetCompatibleCall<megamol::probe::CallProbesDescription>();
    //this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_event_slot.SetCompatibleCall<core::CallEventDescription>();
    this->MakeSlotAvailable(&this->m_event_slot);

    this->m_shading_mode_slot << new megamol::core::param::EnumParam(0);
    this->m_shading_mode_slot.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Grey");
    this->m_shading_mode_slot.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "ClusterID");
    this->MakeSlotAvailable(&this->m_shading_mode_slot);

    this->m_hull_color_slot << new megamol::core::param::ColorParam(
        this->m_hull_color[0], this->m_hull_color[1], this->m_hull_color[2], 1.0f);
    this->MakeSlotAvailable(&this->m_hull_color_slot);
}

megamol::probe_gl::ProbeHullRenderTasks::~ProbeHullRenderTasks() {}

bool megamol::probe_gl::ProbeHullRenderTasks::getDataCallback(core::Call& caller) {
    mesh_gl::CallGPURenderTaskData* lhs_rtc = dynamic_cast<mesh_gl::CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL)
        return false;

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


    if (mc != nullptr) {
        if (!(*mc)(0))
            return false;

        if (m_shading_mode_slot.IsDirty()) {
            m_shading_mode_slot.ResetDirty();

            struct PerFrameData {
                int shading_mode;
            };

            std::array<PerFrameData, 1> per_frame_data;
            per_frame_data[0].shading_mode = m_shading_mode_slot.Param<core::param::EnumParam>()->Value();

            m_rendertask_collection.first->updatePerFrameDataBuffer("", per_frame_data, 1);
        }

        if (m_hull_color_slot.IsDirty()) {
            m_hull_color_slot.ResetDirty();

            std::array<float, 4> obj_color = this->m_hull_color_slot.Param<core::param::ColorParam>()->Value();

            for (auto& batch : m_per_object_data) {
                for (auto& data : batch) {
                    data.color = obj_color;
                }
            }

            for (int i = 0; i < m_batch_meshes.size(); ++i) {
                for (int j = 0; j < m_identifiers[i].size(); ++j) {
                    m_rendertask_collection.first->updatePerDrawData(
                        m_identifiers[i][j], std::vector<PerObjectData>{m_per_object_data[i][j]});
                }
            }
        }

        bool something_has_changed = mc->hasUpdate();

        if (something_has_changed) {
            ++m_version;

            for (auto& identifier : m_rendertask_collection.second) {
                m_rendertask_collection.first->deleteRenderTask(identifier);
            }
            m_rendertask_collection.second.clear();

            m_identifiers.clear();
            m_draw_commands.clear();
            m_per_object_data.clear();
            m_batch_meshes.clear();

            auto gpu_mesh_storage = mc->getData();

            for (auto& mesh_collection : *gpu_mesh_storage) {

                std::shared_ptr<glowl::Mesh> prev_mesh(nullptr);

                int counter = 0;
                for (auto& sub_mesh : mesh_collection->getSubMeshData()) {
                    auto const& gpu_batch_mesh = sub_mesh.second.mesh->mesh;

                    //counter++;
                    //if (counter == 4) {
                    //    continue;
                    //}

                    if (gpu_batch_mesh != prev_mesh) {
                        m_identifiers.emplace_back(std::vector<std::string>());
                        m_draw_commands.emplace_back(std::vector<glowl::DrawElementsCommand>());
                        m_per_object_data.emplace_back(std::vector<PerObjectData>());
                        m_batch_meshes.push_back(gpu_batch_mesh);

                        prev_mesh = gpu_batch_mesh;
                    }

                    float scale = 1.0f;
                    std::array<float, 16> obj_xform = {scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f,
                        scale, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
                    std::array<float, 4> obj_color = this->m_hull_color_slot.Param<core::param::ColorParam>()->Value();

                    m_identifiers.back().emplace_back(std::string(FullName()) + "_" + sub_mesh.first);
                    m_draw_commands.back().push_back(sub_mesh.second.sub_mesh_draw_command);
                    m_per_object_data.back().push_back(PerObjectData{obj_xform, obj_color});
                }
            }

            if (m_show_hull) {
                auto patch_shader = m_material_collection->getMaterial("ProbeHull").shader_program;
                auto tri_shader = m_material_collection->getMaterial("ProbeTriangleHull").shader_program;

                for (int i = 0; i < m_batch_meshes.size(); ++i) {

                    if (m_batch_meshes[i]->getPrimitiveType() == GL_TRIANGLES) {
                        m_rendertask_collection.first->addRenderTasks(
                            m_identifiers[i], tri_shader, m_batch_meshes[i], m_draw_commands[i], m_per_object_data[i]);
                        m_rendertask_collection.second.insert(
                            m_rendertask_collection.second.end(), m_identifiers[i].begin(), m_identifiers[i].end());
                    } else if (m_batch_meshes[i]->getPrimitiveType() == GL_PATCHES) {
                        m_rendertask_collection.first->addRenderTasks(m_identifiers[i], patch_shader, m_batch_meshes[i],
                            m_draw_commands[i], m_per_object_data[i]);
                        m_rendertask_collection.second.insert(
                            m_rendertask_collection.second.end(), m_identifiers[i].begin(), m_identifiers[i].end());
                    } else {
                        //TODO print warning
                    }
                }
            }
        }

        // check for pending events
        auto call_event_storage = this->m_event_slot.CallAs<core::CallEvent>();
        if (call_event_storage != NULL) {
            if ((!(*call_event_storage)(0)))
                return false;

            auto event_collection = call_event_storage->getData();

            // process toggle show glyph events
            {
                auto pending_deselect_events = event_collection->get<ToggleShowHull>();
                for (auto& evt : pending_deselect_events) {
                    m_show_hull = !m_show_hull;

                    if (m_show_hull) {
                        //TODO get rid of code copy-pasting...
                        auto patch_shader = m_material_collection->getMaterial("ProbeHull").shader_program;
                        auto tri_shader = m_material_collection->getMaterial("ProbeTriangleHull").shader_program;

                        for (int i = 0; i < m_batch_meshes.size(); ++i) {

                            if (m_batch_meshes[i]->getPrimitiveType() == GL_TRIANGLES) {
                                m_rendertask_collection.first->addRenderTasks(m_identifiers[i], tri_shader,
                                    m_batch_meshes[i], m_draw_commands[i], m_per_object_data[i]);
                                m_rendertask_collection.second.insert(m_rendertask_collection.second.end(),
                                    m_identifiers[i].begin(), m_identifiers[i].end());
                            } else if (m_batch_meshes[i]->getPrimitiveType() == GL_PATCHES) {
                                m_rendertask_collection.first->addRenderTasks(m_identifiers[i], patch_shader,
                                    m_batch_meshes[i], m_draw_commands[i], m_per_object_data[i]);
                                m_rendertask_collection.second.insert(m_rendertask_collection.second.end(),
                                    m_identifiers[i].begin(), m_identifiers[i].end());
                            } else {
                                // TODO print warning
                            }
                        }
                    } else {
                        for (auto& identifier : m_rendertask_collection.second) {
                            m_rendertask_collection.first->deleteRenderTask(identifier);
                        }
                        m_rendertask_collection.second.clear();
                    }
                }
            }
        }

        // TODO merge meta data stuff, i.e. bounding box
        auto mesh_meta_data = mc->getMetaData();
    }

    // set data if necessary
    lhs_rtc->setData(gpu_render_tasks, m_version);

    return true;
}

bool megamol::probe_gl::ProbeHullRenderTasks::getMetaDataCallback(core::Call& caller) {

    if (!AbstractGPURenderTaskDataSource::getMetaDataCallback(caller))
        return false;

    mesh_gl::CallGPURenderTaskData* lhs_rt_call = dynamic_cast<mesh_gl::CallGPURenderTaskData*>(&caller);
    mesh_gl::CallGPUMeshData* mesh_call = this->m_mesh_slot.CallAs<mesh_gl::CallGPUMeshData>();

    auto lhs_meta_data = lhs_rt_call->getMetaData();

    if (mesh_call != NULL) {
        auto mesh_meta_data = mesh_call->getMetaData();
        mesh_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
        mesh_call->setMetaData(mesh_meta_data);
        if (!(*mesh_call)(1))
            return false;
        mesh_meta_data = mesh_call->getMetaData();

        auto bbox = lhs_meta_data.m_bboxs.BoundingBox();
        auto cbbox = lhs_meta_data.m_bboxs.ClipBox();

        auto mesh_bbox = mesh_meta_data.m_bboxs.BoundingBox();
        auto mesh_cbbox = mesh_meta_data.m_bboxs.ClipBox();

        // mesh_bbox.SetSize(vislib::math::Dimension<float, 3>(
        //    2.1f * mesh_bbox.GetSize()[0], 2.1f * mesh_bbox.GetSize()[1], 2.1f * mesh_bbox.GetSize()[2]));

        mesh_cbbox.SetSize(vislib::math::Dimension<float, 3>(
            2.1f * mesh_cbbox.GetSize()[0], 2.1f * mesh_cbbox.GetSize()[1], 2.1f * mesh_cbbox.GetSize()[2]));

        bbox.Union(mesh_bbox);
        cbbox.Union(mesh_cbbox);

        lhs_meta_data.m_bboxs.SetBoundingBox(bbox);
        lhs_meta_data.m_bboxs.SetClipBox(cbbox);
    }

    lhs_rt_call->setMetaData(lhs_meta_data);

    return true;
}
