#include "ProbeBillboardGlyphRenderTasks.h"

#include "mesh/MeshCalls.h"
#include "ProbeCalls.h"

#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::ProbeBillboardGlyphRenderTasks() 
    : m_probes_slot("GetProbes", "Slot for accessing a probe collection"), m_probes_cached_hash(0), m_billboard_dummy_mesh(nullptr) {

    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

}

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::~ProbeBillboardGlyphRenderTasks() {}

bool megamol::probe_gl::ProbeBillboardGlyphRenderTasks::getDataCallback(core::Call& caller) { 

    mesh::CallGPURenderTaskData* lhs_rtc = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL) return false;

    // no incoming render task collection -> use your own collection
    if (lhs_rtc->getData() == nullptr) lhs_rtc->setData(this->m_gpu_render_tasks);
    std::shared_ptr<mesh::GPURenderTaskCollection> rt_collection = lhs_rtc->getData();

    // if there is a render task connection to the right, pass on the render task collection
    mesh::CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<mesh::CallGPURenderTaskData>();
    if (rhs_rtc != NULL) {
        rhs_rtc->setData(rt_collection);
        if (!(*rhs_rtc)(0)) return false;
    }

    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
    if (mtlc == NULL) return false;
    if (!(*mtlc)(0)) return false;
    auto gpu_mtl_storage = mtlc->getData();


    // create an empty dummy mesh, actual billboard geometry will be build in vertex shader
    if (m_billboard_dummy_mesh == nullptr) {

        std::vector<void*> data_ptrs = {nullptr};
        std::vector<size_t> byte_sizes = {0};
        std::vector<uint32_t> indices = {0,1,2,3,4,5}; 
        glowl::VertexLayout vertex_layout;

        m_billboard_dummy_mesh = std::make_shared<glowl::Mesh>(
            data_ptrs, byte_sizes, indices.data(), 6*4, vertex_layout, GL_UNSIGNED_INT, GL_STATIC_DRAW, GL_TRIANGLES);
    }

    probe::CallProbes* pc = this->m_probes_slot.CallAs<probe::CallProbes>();
    if (pc == NULL) return false;

    auto probe_meta_data = pc->getMetaData();

    if (probe_meta_data.m_data_hash > m_probes_cached_hash) {
        m_probes_cached_hash = probe_meta_data.m_data_hash;

        if (!(*pc)(0)) return false;
        auto probes = pc->getData();

        auto probe_cnt = probes->getProbeCount();

        std::vector<glowl::DrawElementsCommand> draw_commands;
        std::vector<glm::vec4> glpyh_positions;

        draw_commands.reserve(probe_cnt);
        glpyh_positions.reserve(probe_cnt);

        for (int probe_idx = 0; probe_idx < probe_cnt; ++probe_idx) {
            try {
                auto probe = probes->getProbe<probe::FloatProbe>(probe_idx);

                glpyh_positions.push_back(glm::vec4(
                    probe.m_position[0] + probe.m_direction[0] * probe.m_begin, 
                    probe.m_position[1] + probe.m_direction[1] * probe.m_begin, 
                    probe.m_position[2] + probe.m_direction[2] * probe.m_begin,
                    1.0f));

                glowl::DrawElementsCommand draw_command;
                draw_command.base_instance = 0;
                draw_command.base_vertex = 0;
                draw_command.cnt = 6;
                draw_command.first_idx = 0;
                draw_command.instance_cnt = 1;

                draw_commands.push_back(draw_command);

            } catch (std::bad_variant_access&) {
                // TODO log error, dont add new render task
            }
        }

        auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;
        rt_collection->addRenderTasks(shader, m_billboard_dummy_mesh, draw_commands, glpyh_positions);
    }


    return true; 
}

bool megamol::probe_gl::ProbeBillboardGlyphRenderTasks::getMetaDataCallback(core::Call& caller) {

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
