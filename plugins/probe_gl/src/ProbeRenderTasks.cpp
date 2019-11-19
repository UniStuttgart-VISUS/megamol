#include "ProbeRenderTasks.h"

#include "ProbeCalls.h"
#include "mesh/MeshCalls.h"
#include "ProbeGlCalls.h"

#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"

megamol::probe_gl::ProbeRenderTasks::ProbeRenderTasks()
    : m_probes_slot("GetProbes", "Slot for accessing a probe collection")
    , m_probes_cached_hash(0)
    , m_probe_manipulation_slot("GetProbeManipulation", "") 
{
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


    // no incoming render task collection -> use your own collection
    if (lhs_rtc->getData() == nullptr) lhs_rtc->setData(this->m_gpu_render_tasks);
    std::shared_ptr<mesh::GPURenderTaskCollection> rt_collection = lhs_rtc->getData();

    // if there is a render task connection to the right, pass on the render task collection
    mesh::CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<mesh::CallGPURenderTaskData>();
    if (rhs_rtc != NULL) {
        rhs_rtc->setData(rt_collection);
        if (!(*rhs_rtc)(0)) return false;
    }

    auto gpu_mtl_storage = mtlc->getData();
    auto gpu_mesh_storage = mc->getData();

    auto probe_meta_data = pc->getMetaData();

    if (probe_meta_data.m_data_hash > m_probes_cached_hash) {
        m_probes_cached_hash = probe_meta_data.m_data_hash;

        if (!(*pc)(0)) return false;
        auto probes = pc->getData();

        auto probe_cnt = probes->getProbeCount();

        for (int probe_idx = 0; probe_idx < probe_cnt; ++probe_idx) {
            try {
                auto probe = probes->getProbe<probe::FloatProbe>(probe_idx);

                // TODO create and add new render task for probe

                assert(gpu_mesh_storage->getSubMeshData().size() > 0);

                auto const& gpu_sub_mesh = gpu_mesh_storage->getSubMeshData().front();
                auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes().front().mesh;
                auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

                std::vector<glowl::DrawElementsCommand> draw_commands(1, gpu_sub_mesh.sub_mesh_draw_command);

                //std::vector<std::array<float, 16>> object_transform(1);
                std::array<glm::mat4x4,1> object_transform;

                const glm::vec3 from(0.0f, 1.0f, 0.0f);
                const glm::vec3 to(probe.m_direction[0], probe.m_direction[1], probe.m_direction[2]);
                glm::vec3 v = glm::cross(to, from);
                float angle = -acos(glm::dot(to, from) / (glm::length(to) * glm::length(from)));
                std::get<0>(object_transform) = glm::rotate(angle, v);

                auto scaling = glm::scale(glm::vec3(0.5f, probe.m_end - probe.m_begin, 0.5f));

                auto probe_start_point = glm::vec3(
                    probe.m_position[0] + probe.m_direction[0] * probe.m_begin,
                    probe.m_position[1] + probe.m_direction[1] * probe.m_begin,
                    probe.m_position[2] + probe.m_direction[2] * probe.m_begin);
                auto translation = glm::translate(glm::mat4(), probe_start_point);
                std::get<0>(object_transform) = translation * std::get<0>(object_transform) * scaling;

                

                rt_collection->addRenderTasks(shader, gpu_batch_mesh, draw_commands, object_transform);

            } catch (std::bad_variant_access&) {
                // TODO log error, dont add new render task
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

    if (probe_meta_data.m_data_hash > m_probes_cached_hash) lhs_meta_data.m_data_hash++;
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
