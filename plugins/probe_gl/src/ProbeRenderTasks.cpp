#include "ProbeRenderTasks.h"

#include "mmcore/EventCall.h"

#include "ProbeEvents.h"
#include "ProbeGlCalls.h"
#include "mesh/MeshCalls.h"
#include "probe/ProbeCalls.h"

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/transform.hpp"

megamol::probe_gl::ProbeRenderTasks::ProbeRenderTasks()
        : m_version(0)
        , m_material_collection(nullptr)
        , m_show_probes(true)
        , m_probes_slot("GetProbes", "Slot for accessing a probe collection")
        , m_event_slot("GetEvents", "") {
    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_event_slot.SetCompatibleCall<megamol::core::CallEventDescription>();
    this->MakeSlotAvailable(&this->m_event_slot);
}

megamol::probe_gl::ProbeRenderTasks::~ProbeRenderTasks() {}

bool megamol::probe_gl::ProbeRenderTasks::create() {
    auto retval = AbstractGPURenderTaskDataSource::create();

    m_material_collection = std::make_shared<mesh_gl::GPUMaterialCollection>();
    try {
        std::vector<std::filesystem::path> shaderfiles = {
            "probes/dfr_interaction_probe.vert.glsl", "probes/dfr_interaction_probe.frag.glsl"};
        m_material_collection->addMaterial(this->instance(), "ProbeInteraction", shaderfiles);
    } catch (std::runtime_error const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "%s [%s, %s, line %d]\n", ex.what(), __FILE__, __FUNCTION__, __LINE__);
        retval = false;
    }

    return retval;
}

bool megamol::probe_gl::ProbeRenderTasks::getDataCallback(core::Call& caller) {
    {
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("GL error during rendering" + err);
        }
    }

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
    if (mc == NULL)
        return false;
    if (!(*mc)(0))
        return false; // TODO only call callback when hash is outdated?

    probe::CallProbes* pc = this->m_probes_slot.CallAs<probe::CallProbes>();
    if (pc == NULL)
        return false;
    if (!(*pc)(0))
        return false;

    // something has changed in the neath
    bool something_has_changed = mc->hasUpdate() || pc->hasUpdate();

    auto gpu_mesh_storage = mc->getData();
    auto probes = pc->getData();

    struct PerObjData {
        glm::mat4x4 object_transform;
        int highlighted;
        float pad0;
        float pad1;
        float pad2;
    };

    if (something_has_changed) {
        ++m_version;

        for (auto& identifier : m_rendertask_collection.second) {
            m_rendertask_collection.first->deleteRenderTask(identifier);
        }
        m_rendertask_collection.second.clear();

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

                assert(gpu_mesh_storage->front()->getSubMeshData().size() > 0);

                m_identifiers[probe_idx] = (std::string(FullName()) + "_probe_" + std::to_string(probe_idx));

                auto const& gpu_sub_mesh = gpu_mesh_storage->front()->getSubMeshData().begin()->second;

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

        auto const& gpu_sub_mesh = gpu_mesh_storage->front()->getSubMeshData().begin()->second;
        auto const& shader = m_material_collection->getMaterials().begin()->second.shader_program;

        if (m_show_probes) {
            m_rendertask_collection.first->addRenderTasks(
                m_identifiers, shader, gpu_sub_mesh.mesh->mesh, m_draw_commands, m_probe_draw_data);
            m_rendertask_collection.second.insert(
                m_rendertask_collection.second.end(), m_identifiers.begin(), m_identifiers.end());
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
        if ((!(*call_event_storage)(0)))
            return false;

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
                        m_rendertask_collection.first->updatePerDrawData(m_identifiers[i], per_probe_data);
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
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
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
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
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
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
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
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                }
            }
        }

        // process toggle show glyph events
        {
            auto pending_deselect_events = event_collection->get<ToggleShowProbes>();
            for (auto& evt : pending_deselect_events) {
                m_show_probes = !m_show_probes;

                if (m_show_probes) {
                    auto const& gpu_sub_mesh = gpu_mesh_storage->front()->getSubMeshData().begin()->second;
                    auto const& shader = m_material_collection->getMaterials().begin()->second.shader_program;
                    m_rendertask_collection.first->addRenderTasks(
                        m_identifiers, shader, gpu_sub_mesh.mesh->mesh, m_draw_commands, m_probe_draw_data);
                    m_rendertask_collection.second.insert(
                        m_rendertask_collection.second.end(), m_identifiers.begin(), m_identifiers.end());
                } else {
                    for (auto& identifier : m_rendertask_collection.second) {
                        m_rendertask_collection.first->deleteRenderTask(identifier);
                    }
                    m_rendertask_collection.second.clear();
                }
            }
        }
    }

    lhs_rtc->setData(gpu_render_tasks, m_version);

    return true;
}

bool megamol::probe_gl::ProbeRenderTasks::getMetaDataCallback(core::Call& caller) {

    if (!AbstractGPURenderTaskDataSource::getMetaDataCallback(caller))
        return false;

    mesh_gl::CallGPURenderTaskData* lhs_rt_call = dynamic_cast<mesh_gl::CallGPURenderTaskData*>(&caller);
    auto probe_call = m_probes_slot.CallAs<probe::CallProbes>();
    if (probe_call == NULL)
        return false;

    auto lhs_meta_data = lhs_rt_call->getMetaData();

    auto probe_meta_data = probe_call->getMetaData();
    probe_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
    probe_call->setMetaData(probe_meta_data);
    if (!(*probe_call)(1))
        return false;
    probe_meta_data = probe_call->getMetaData();

    lhs_meta_data.m_frame_cnt = std::min(lhs_meta_data.m_frame_cnt, probe_meta_data.m_frame_cnt);

    auto bbox = lhs_meta_data.m_bboxs.BoundingBox();
    bbox.Union(probe_meta_data.m_bboxs.BoundingBox());
    lhs_meta_data.m_bboxs.SetBoundingBox(bbox);

    auto cbbox = lhs_meta_data.m_bboxs.ClipBox();
    cbbox.Union(probe_meta_data.m_bboxs.ClipBox());
    lhs_meta_data.m_bboxs.SetClipBox(cbbox);

    lhs_rt_call->setMetaData(lhs_meta_data);

    return true;
}
