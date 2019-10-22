#include "ProbeBillboardGlyphRenderTasks.h"

#include "mesh/MeshCalls.h"
#include "ProbeCalls.h"

#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::ProbeBillboardGlyphRenderTasks() 
    : m_probes_slot("GetProbes", "Slot for accessing a probe collection"), m_probes_cached_hash(0) {

    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

}

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::~ProbeBillboardGlyphRenderTasks() {}

bool megamol::probe_gl::ProbeBillboardGlyphRenderTasks::getDataCallback(core::Call& caller) { 

    mesh::CallGPURenderTaskData* lhs_mtl_call = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    if (lhs_mtl_call == NULL) return false;

    // no incoming render task collection -> use your own collection
    if (lhs_mtl_call->getData() == nullptr) lhs_mtl_call->setData(this->m_gpu_render_tasks);
    std::shared_ptr<mesh::GPURenderTaskCollection> rt_collection = lhs_mtl_call->getData();

    // if there is a render task connection to the right, pass on the render task collection
    mesh::CallGPURenderTaskData* rhs_mtl_call = this->m_renderTask_rhs_slot.CallAs<mesh::CallGPURenderTaskData>();
    if (rhs_mtl_call != NULL) rhs_mtl_call->setData(rt_collection);


    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
    if (mtlc == NULL) return false;
    if (!(*mtlc)(0)) return false;
    auto gpu_mtl_storage = mtlc->getData();

    probe::CallProbes* pc = this->m_probes_slot.CallAs<probe::CallProbes>();
    if (pc == NULL) return false;

    auto probe_meta_data = pc->getMetaData();

    if (probe_meta_data.m_data_hash > m_probes_cached_hash) {
        m_probes_cached_hash = probe_meta_data.m_data_hash;

        if (!(*pc)(0)) return false;
        auto probes = pc->getData();

        auto probe_cnt = probes->getProbeCount();

        for (int probe_idx = 0; probe_idx < probe_cnt; ++probe_idx) {
            try {
                auto probe = probes->getProbe<probe::FloatProbe>(probe_idx);

                auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

                //std::vector<glowl::DrawElementsCommand> draw_commands(1, gpu_sub_mesh.sub_mesh_draw_command);

                // std::vector<std::array<float, 16>> object_transform(1);
                std::array<glm::mat4x4, 1> object_transform;
                auto scaling = glm::scale(glm::vec3(0.5f, probe.m_end - probe.m_begin, 0.5f));

                auto translation = glm::translate(
                    glm::mat4(), glm::vec3(probe.m_position[0], probe.m_position[1], probe.m_position[2]));
                std::get<0>(object_transform) = translation * std::get<0>(object_transform) * scaling;

                //m_gpu_render_tasks->addRenderTasks(shader, gpu_batch_mesh, draw_commands, object_transform);

            } catch (std::bad_variant_access&) {
                // TODO log error, dont add new render task
            }
        }
    }


    return true; 
}

bool megamol::probe_gl::ProbeBillboardGlyphRenderTasks::getMetaDataCallback(core::Call& caller) {

    if (!AbstractGPURenderTaskDataSource::getMetaDataCallback(caller)) return false;

    mesh::CallGPURenderTaskData* lhs_rt_call = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    auto probe_call = m_probes_slot.CallAs<probe::CallProbes>();

    auto lhs_meta_data = lhs_rt_call->getMetaData();

    auto probe_meta_data = probe_call->getMetaData();
    probe_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
    probe_call->setMetaData(probe_meta_data);
    if (!(*probe_call)(1)) return false;
    probe_meta_data = probe_call->getMetaData();


    if (probe_meta_data.m_data_hash > m_probes_cached_hash) {
        m_renderTask_lhs_cached_hash++;
    }

    lhs_meta_data.m_data_hash = m_renderTask_lhs_cached_hash;
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
