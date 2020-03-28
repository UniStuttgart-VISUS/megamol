#include "ProbeRenderTasks.h"

#include "ProbeCalls.h"
#include "ProbeGlCalls.h"
#include "mesh/MeshCalls.h"

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/transform.hpp"

megamol::probe_gl::ProbeRenderTasks::ProbeRenderTasks()
    : m_version(0)
    , m_probes_slot("GetProbes", "Slot for accessing a probe collection")
    , m_probe_manipulation_slot("GetProbeManipulation", "") {
    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_probe_manipulation_slot.SetCompatibleCall<probe_gl::CallProbeInteractionDescription>();
    this->MakeSlotAvailable(&this->m_probe_manipulation_slot);
}

megamol::probe_gl::ProbeRenderTasks::~ProbeRenderTasks() {}

bool megamol::probe_gl::ProbeRenderTasks::getDataCallback(core::Call& caller) {

    mesh::CallGPURenderTaskData* lhs_rtc = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL) return false;

    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
    if (mtlc == NULL) return false;
    if (!(*mtlc)(0)) return false;

    mesh::CallGPUMeshData* mc = this->m_mesh_slot.CallAs<mesh::CallGPUMeshData>();
    if (mc == NULL) return false;
    if (!(*mc)(0)) return false; // TODO only call callback when hash is outdated?

    probe::CallProbes* pc = this->m_probes_slot.CallAs<probe::CallProbes>();
    if (pc == NULL) return false;
    if (!(*pc)(0)) return false;

    // something has changed in the neath
    bool something_has_changed = mtlc->hasUpdate() || mc->hasUpdate() || pc->hasUpdate();

    // no incoming render task collection -> use your own collection
    std::shared_ptr<mesh::GPURenderTaskCollection> rt_collection;
    if (lhs_rtc->getData() == nullptr) rt_collection = this->m_gpu_render_tasks;
    else rt_collection = lhs_rtc->getData();

    // if there is a render task connection to the right, pass on the render task collection
    mesh::CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<mesh::CallGPURenderTaskData>();
    if (rhs_rtc != NULL) {
        rhs_rtc->setData(rt_collection,0);
        if (!(*rhs_rtc)(0)) return false;
    }

    struct PerObjData {
        glm::mat4x4 object_transform;
        int highlighted;
        float pad0;
        float pad1;
        float pad2;
    };

    if (something_has_changed) {
        ++m_version;

        //TODO this breaks chaining...
        rt_collection->clear();

        auto gpu_mtl_storage = mtlc->getData();
        auto gpu_mesh_storage = mc->getData();
        auto probes = pc->getData();

        auto probe_cnt = probes->getProbeCount();

        m_probe_draw_data.clear();
        m_probe_draw_data.resize(probe_cnt);

        std::vector<glowl::DrawElementsCommand> draw_commands(probe_cnt);

        for (int probe_idx = 0; probe_idx < probe_cnt; ++probe_idx) {
            try {
                //auto probe = probes->getProbe<probe::FloatProbe>(probe_idx);

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

                assert(gpu_mesh_storage->getSubMeshData().size() > 0);

                auto const& gpu_sub_mesh = gpu_mesh_storage->getSubMeshData().front();
                auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes().front().mesh;

                draw_commands[probe_idx] = gpu_sub_mesh.sub_mesh_draw_command;

                const glm::vec3 from(0.0f, 0.0f, 1.0f);
                const glm::vec3 to(direction[0], direction[1], direction[2]);
                glm::vec3 v = glm::cross(to, from);
                float angle = -acos(glm::dot(to, from) / (glm::length(to) * glm::length(from)));
                m_probe_draw_data[probe_idx].object_transform = glm::rotate(angle, v);

                auto scaling = glm::scale(glm::vec3(0.5f, 0.5f, end - begin));

                auto probe_start_point = glm::vec3(
                    position[0] + direction[0] * begin,
                    position[1] + direction[1] * begin,
                    position[2] + direction[2] * begin);
                auto translation = glm::translate(glm::mat4(), probe_start_point);
                m_probe_draw_data[probe_idx].object_transform =
                    translation * m_probe_draw_data[probe_idx].object_transform * scaling;

            } catch (std::bad_variant_access&) {
                // TODO log error, dont add new render task
            }
        }

        auto const& gpu_sub_mesh = gpu_mesh_storage->getSubMeshData().front();
        auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes().front().mesh;
        auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;
        rt_collection->addRenderTasks(shader, gpu_batch_mesh, draw_commands, m_probe_draw_data);
    }

    lhs_rtc->setData(rt_collection, m_version);

    // check for pending probe manipulations
    CallProbeInteraction* pic = this->m_probe_manipulation_slot.CallAs<CallProbeInteraction>();
    if (pic != NULL) {
        if (!(*pic)(0)) return false;

        if (pic->hasUpdate()) {
            auto interaction_collection = pic->getData();

            auto& pending_manips = interaction_collection->accessPendingManipulations();

            if (pc->hasUpdate())
            {
                if (!(*pc)(0)) return false;
            }
            auto probes = pc->getData();

            for (auto itr = pending_manips.begin(); itr != pending_manips.end(); ++itr) {
                if (itr->type == HIGHLIGHT) 
                {
                    // TODO remove from list and apply hightlight to render task
                    auto manipulation = *itr;
                    //itr = pending_manips.erase(itr);

                    std::array<PerProbeDrawData, 1> per_probe_data = {m_probe_draw_data[manipulation.obj_id]};
                    per_probe_data[0].highlighted = 1;

                    rt_collection->updatePerDrawData(manipulation.obj_id, per_probe_data);
                }
                else if (itr->type == DEHIGHLIGHT)
                {
                    // TODO remove from list and apply hightlight to render task
                    auto manipulation = *itr;
                    //itr = pending_manips.erase(itr);

                    std::array<PerProbeDrawData,1> per_probe_data = { m_probe_draw_data[manipulation.obj_id] };

                    rt_collection->updatePerDrawData(manipulation.obj_id, per_probe_data);
                } 
                else if (itr->type == SELECT) 
                {
                    // TODO remove from list and apply hightlight to render task
                    auto manipulation = *itr;
                    //itr = pending_manips.erase(itr);

                    m_probe_draw_data[manipulation.obj_id].highlighted = 2;
                    std::array<PerProbeDrawData, 1> per_probe_data = {m_probe_draw_data[manipulation.obj_id]};

                    rt_collection->updatePerDrawData(manipulation.obj_id, per_probe_data);
                }
                else {
                    ++itr;
                }
            }
        }
    }

    return true;
}

bool megamol::probe_gl::ProbeRenderTasks::getMetaDataCallback(core::Call& caller) {

    if (!AbstractGPURenderTaskDataSource::getMetaDataCallback(caller)) return false;

    mesh::CallGPURenderTaskData* lhs_rt_call = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    auto probe_call = m_probes_slot.CallAs<probe::CallProbes>();
    if (probe_call == NULL) return false;

    auto lhs_meta_data = lhs_rt_call->getMetaData();

    auto probe_meta_data = probe_call->getMetaData();
    probe_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
    probe_call->setMetaData(probe_meta_data);
    if (!(*probe_call)(1)) return false;
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
