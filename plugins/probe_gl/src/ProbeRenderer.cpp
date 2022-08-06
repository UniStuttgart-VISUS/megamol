#include "ProbeRenderer.h"

#include "mmstd/event/EventCall.h"

#include "ProbeEvents.h"
#include "ProbeGlCalls.h"
#include "mesh/MeshCalls.h"
#include "probe/ProbeCalls.h"

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/transform.hpp"

megamol::probe_gl::ProbeRenderTasks::ProbeRenderTasks()
        : m_show_probes(true)
        , m_probes_slot("GetProbes", "Slot for accessing a probe collection")
        , m_event_slot("GetEvents", "") {
    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_event_slot.SetCompatibleCall<megamol::core::CallEventDescription>();
    this->MakeSlotAvailable(&this->m_event_slot);
}

megamol::probe_gl::ProbeRenderTasks::~ProbeRenderTasks() {}

bool megamol::probe_gl::ProbeRenderTasks::GetExtents(mmstd_gl::CallRender3DGL& call) {
    bool retval = false;
    auto probe_call = m_probes_slot.CallAs<probe::CallProbes>();
    if (probe_call != nullptr) {
        auto probe_meta_data = probe_call->getMetaData();
        probe_meta_data.m_frame_ID = static_cast<int>(call.Time());
        probe_call->setMetaData(probe_meta_data);
        if ((*probe_call)(1)) {
            probe_meta_data = probe_call->getMetaData();

            call.SetTimeFramesCount(std::min(call.TimeFramesCount(), probe_meta_data.m_frame_cnt));

            auto bbox = call.AccessBoundingBoxes().BoundingBox();
            bbox.Union(probe_meta_data.m_bboxs.BoundingBox());
            call.AccessBoundingBoxes().SetBoundingBox(bbox);

            auto cbbox = call.AccessBoundingBoxes().ClipBox();
            cbbox.Union(probe_meta_data.m_bboxs.ClipBox());
            call.AccessBoundingBoxes().SetClipBox(cbbox);

            retval = true;
        }
    }
    return retval;
}

void megamol::probe_gl::ProbeRenderTasks::createMaterialCollection() {
    material_collection_ = std::make_shared<mesh_gl::GPUMaterialCollection>();
    try {
        material_collection_->addMaterial(this->instance(), "ProbeInteraction",
            {"probes/dfr_interaction_probe.vert.glsl", "probes/dfr_interaction_probe.frag.glsl"});
    } catch (std::runtime_error const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "%s [%s, %s, line %d]\n", ex.what(), __FILE__, __FUNCTION__, __LINE__);
    }
}

void megamol::probe_gl::ProbeRenderTasks::updateRenderTaskCollection(
    mmstd_gl::CallRender3DGL& call, bool force_update) {
    {
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("GL error during rendering" + err);
        }
    }

    probe::CallProbes* pc = this->m_probes_slot.CallAs<probe::CallProbes>();
    if (pc != NULL) {
        if (!(*pc)(0)) {
            // TODO throw error
            return;
        }

        // something has changed in the neath
        bool something_has_changed = force_update || pc->hasUpdate();

        auto probes = pc->getData();

        struct PerObjData {
            glm::mat4x4 object_transform;
            int highlighted;
            float pad0;
            float pad1;
            float pad2;
        };

        if (something_has_changed) {
            render_task_collection_->clear();

            auto probe_cnt = probes->getProbeCount();

            m_identifiers.clear();
            m_identifiers.resize(probe_cnt);

            m_probe_draw_data.clear();
            m_probe_draw_data.resize(probe_cnt);

            m_draw_commands.clear();
            m_draw_commands.resize(probe_cnt);

            for (int probe_idx = 0; probe_idx < probe_cnt; ++probe_idx) {
                try {
                    // auto probe = probes->getProbe<probe::FloatProbe>(probe_idx);

                    auto generic_probe = probes->getGenericProbe(probe_idx);

                    std::array<float, 3> direction;
                    std::array<float, 3> position;
                    float begin;
                    float end;

                    auto visitor = [&direction, &position, &begin, &end](auto&& arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, probe::FloatProbe>) {
                            direction = arg.m_direction;
                            position = arg.m_position;
                            begin = arg.m_begin;
                            end = arg.m_end;
                        } else if constexpr (std::is_same_v<T, probe::IntProbe>) {
                            direction = arg.m_direction;
                            position = arg.m_position;
                            begin = arg.m_begin;
                            end = arg.m_end;
                        } else if constexpr (std::is_same_v<T, probe::Vec4Probe>) {
                            direction = arg.m_direction;
                            position = arg.m_position;
                            begin = arg.m_begin;
                            end = arg.m_end;
                        } else {
                            // unknown probe type, throw error? do nothing?
                        }
                    };

                    std::visit(visitor, generic_probe);

                    // TODO create and add new render task for probe

                    assert(mesh_collection_->getSubMeshData().size() > 0);

                    m_identifiers[probe_idx] = (std::string(FullName()) + "_probe_" + std::to_string(probe_idx));

                    auto const& gpu_sub_mesh = mesh_collection_->getSubMeshData().begin()->second;

                    m_draw_commands[probe_idx] = gpu_sub_mesh.sub_mesh_draw_command;

                    const glm::vec3 from(0.0f, 0.0f, 1.0f);
                    const glm::vec3 to(direction[0], direction[1], direction[2]);
                    glm::vec3 v = glm::cross(to, from);
                    float angle = -acos(glm::dot(to, from) / (glm::length(to) * glm::length(from)));
                    m_probe_draw_data[probe_idx].object_transform = glm::rotate(angle, v);

                    auto scaling = glm::scale(glm::vec3(end - begin));

                    auto probe_start_point = glm::vec3(position[0] + direction[0] * begin,
                        position[1] + direction[1] * begin, position[2] + direction[2] * begin);
                    auto translation = glm::translate(glm::mat4(), probe_start_point);
                    m_probe_draw_data[probe_idx].object_transform =
                        translation * m_probe_draw_data[probe_idx].object_transform * scaling;

                } catch (std::bad_variant_access&) {
                    // TODO log error, dont add new render task
                }
            }

            auto const& gpu_sub_mesh = mesh_collection_->getSubMeshData().begin()->second;
            auto const& shader = material_collection_->getMaterial("ProbeInteraction").shader_program;

            if (m_show_probes) {
                render_task_collection_->addRenderTasks(
                    m_identifiers, shader, gpu_sub_mesh.mesh->mesh, m_draw_commands, m_probe_draw_data);
            }
        }

        {
            GLenum err = glGetError();
            if (err != GL_NO_ERROR) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("GL error during rendering" + err);
            }
        }

        // check for pending events
        auto call_event_storage = this->m_event_slot.CallAs<core::CallEvent>();
        if (call_event_storage != NULL) {
            if ((!(*call_event_storage)(0))) {
                // TODO throw error
                return;
            }

            auto event_collection = call_event_storage->getData();

            // process pobe clear selection events
            {
                auto pending_clearselection_events = event_collection->get<ProbeClearSelection>();
                for (auto& evt : pending_clearselection_events) {
                    for (auto& draw_data : m_probe_draw_data) {
                        draw_data.highlighted = 0;
                    }

                    if (m_show_probes) {
                        for (int i = 0; i < m_probe_draw_data.size(); ++i) {
                            std::array<PerProbeDrawData, 1> per_probe_data = {m_probe_draw_data[i]};
                            render_task_collection_->updatePerDrawData(m_identifiers[i], per_probe_data);
                        }
                    }
                }
            }

            // process probe highlight events
            {
                auto pending_highlight_events = event_collection->get<ProbeHighlight>();
                for (auto& evt : pending_highlight_events) {
                    std::array<PerProbeDrawData, 1> per_probe_data = {m_probe_draw_data[evt.obj_id]};
                    per_probe_data[0].highlighted = 1;

                    if (m_show_probes) {
                        std::string identifier = std::string(FullName()) + "_probe_" + std::to_string(evt.obj_id);
                        render_task_collection_->updatePerDrawData(identifier, per_probe_data);
                    }
                }
            }

            // process probe dehighlight events
            {
                auto pending_dehighlight_events = event_collection->get<ProbeDehighlight>();
                for (auto& evt : pending_dehighlight_events) {
                    std::array<PerProbeDrawData, 1> per_probe_data = {m_probe_draw_data[evt.obj_id]};

                    if (m_show_probes) {
                        std::string identifier = std::string(FullName()) + "_probe_" + std::to_string(evt.obj_id);
                        render_task_collection_->updatePerDrawData(identifier, per_probe_data);
                    }
                }
            }

            // process probe selection events
            {
                auto pending_select_events = event_collection->get<ProbeSelect>();
                for (auto& evt : pending_select_events) {
                    m_probe_draw_data[evt.obj_id].highlighted = 2;
                    std::array<PerProbeDrawData, 1> per_probe_data = {m_probe_draw_data[evt.obj_id]};

                    if (m_show_probes) {
                        std::string identifier = std::string(FullName()) + "_probe_" + std::to_string(evt.obj_id);
                        render_task_collection_->updatePerDrawData(identifier, per_probe_data);
                    }
                }
            }

            // process probe deselection events
            {
                auto pending_deselect_events = event_collection->get<ProbeDeselect>();
                for (auto& evt : pending_deselect_events) {
                    m_probe_draw_data[evt.obj_id].highlighted = 0;
                    std::array<PerProbeDrawData, 1> per_probe_data = {m_probe_draw_data[evt.obj_id]};

                    if (m_show_probes) {
                        std::string identifier = std::string(FullName()) + "_probe_" + std::to_string(evt.obj_id);
                        render_task_collection_->updatePerDrawData(identifier, per_probe_data);
                    }
                }
            }

            // process toggle show glyph events
            {
                auto pending_deselect_events = event_collection->get<ToggleShowProbes>();
                for (auto& evt : pending_deselect_events) {
                    m_show_probes = !m_show_probes;

                    if (m_show_probes) {
                        auto const& gpu_sub_mesh = mesh_collection_->getSubMeshData().begin()->second;
                        auto const& shader = material_collection_->getMaterial("ProbeInteraction").shader_program;
                        render_task_collection_->addRenderTasks(
                            m_identifiers, shader, gpu_sub_mesh.mesh->mesh, m_draw_commands, m_probe_draw_data);
                    } else {
                        render_task_collection_->clear();
                    }
                }
            }
        }
    }
}
