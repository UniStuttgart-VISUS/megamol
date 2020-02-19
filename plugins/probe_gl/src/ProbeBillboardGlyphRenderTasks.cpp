#include "ProbeBillboardGlyphRenderTasks.h"

#include "mesh/MeshCalls.h"
#include "ProbeCalls.h"

#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "mmcore/param/FloatParam.h"

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::ProbeBillboardGlyphRenderTasks()
    : m_version(0)
    , m_probes_slot("GetProbes", "Slot for accessing a probe collection")
    , m_billboard_dummy_mesh(nullptr)
    , m_billboard_size_slot("BillBoardSize", "Sets the scaling factor of the texture billboards") {

    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_billboard_size_slot << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->m_billboard_size_slot);
}

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::~ProbeBillboardGlyphRenderTasks() {}

bool megamol::probe_gl::ProbeBillboardGlyphRenderTasks::getDataCallback(core::Call& caller) { 

    mesh::CallGPURenderTaskData* lhs_rtc = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL) return false;

    std::shared_ptr<mesh::GPURenderTaskCollection> rt_collection;
    // no incoming render task collection -> use your own collection
    if (lhs_rtc->getData() == nullptr) 
        rt_collection = this->m_gpu_render_tasks;
    else
        rt_collection = lhs_rtc->getData();

    // if there is a render task connection to the right, pass on the render task collection
    mesh::CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<mesh::CallGPURenderTaskData>();
    if (rhs_rtc != NULL) {
        rhs_rtc->setData(rt_collection,0);
        if (!(*rhs_rtc)(0)) return false;
    }

    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
    if (mtlc == NULL) return false;
    if (!(*mtlc)(0)) return false;
    
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
    if (!(*pc)(0)) return false;

    bool something_has_changed = pc->hasUpdate() || mtlc->hasUpdate() || this->m_billboard_size_slot.IsDirty();

    if (something_has_changed) {
        ++m_version;

        this->m_billboard_size_slot.ResetDirty();
        auto gpu_mtl_storage = mtlc->getData();
        auto probes = pc->getData();
        
        rt_collection->clear();

        auto probe_cnt = probes->getProbeCount();

        std::vector<glowl::DrawElementsCommand> draw_commands;

        struct PerGlyphData {
            glm::vec4 position;
            GLuint64 texture_handle;
            float slice_idx;
            float scale;
        };

        std::vector<PerGlyphData> glyph_data;

        draw_commands.reserve(probe_cnt);
        glyph_data.reserve(probe_cnt);

        for (int probe_idx = 0; probe_idx < probe_cnt; ++probe_idx) {

            auto generic_probe = probes->getGenericProbe(probe_idx);

            auto visitor = [&draw_commands, &glyph_data, &gpu_mtl_storage, probe_idx, this](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, probe::FloatProbe>) {
                    
                    PerGlyphData glyph;
                    glyph.position = glm::vec4(arg.m_position[0] + arg.m_direction[0] * (arg.m_begin * 1.1f),
                        arg.m_position[1] + arg.m_direction[1] * (arg.m_begin * 1.1f),
                        arg.m_position[2] + arg.m_direction[2] * (arg.m_begin * 1.1f), 1.0f);
                    glyph.texture_handle =
                        gpu_mtl_storage->getMaterials().front().textures[probe_idx / 2048]->getTextureHandle();
                    glyph.slice_idx = probe_idx % 2048;
                    glyph.scale = this->m_billboard_size_slot.Param<core::param::FloatParam>()->Value();

                    gpu_mtl_storage->getMaterials().front().textures[probe_idx / 2048]->makeResident();

                    glowl::DrawElementsCommand draw_command;
                    draw_command.base_instance = 0;
                    draw_command.base_vertex = 0;
                    draw_command.cnt = 6;
                    draw_command.first_idx = 0;
                    draw_command.instance_cnt = 1;

                    draw_commands.push_back(draw_command);
                    glyph_data.push_back(glyph);

                } else if constexpr (std::is_same_v<T, probe::IntProbe>) {
                    // TODO
                } else if constexpr (std::is_same_v<T, probe::Vec4Probe>) {

                    PerGlyphData glyph;
                    glyph.position = glm::vec4(arg.m_position[0] + arg.m_direction[0] * (arg.m_begin * 1.1f),
                            arg.m_position[1] + arg.m_direction[1] * (arg.m_begin * 1.1f),
                            arg.m_position[2] + arg.m_direction[2] * (arg.m_begin * 1.1f), 1.0f);
                    glyph.texture_handle = gpu_mtl_storage->getMaterials().front().textures[probe_idx / 2048]->getTextureHandle();
                    glyph.slice_idx = probe_idx % 2048;
                    glyph.scale = this->m_billboard_size_slot.Param<core::param::FloatParam>()->Value();

                    gpu_mtl_storage->getMaterials().front().textures[probe_idx / 2048]->makeResident();

                    glowl::DrawElementsCommand draw_command;
                    draw_command.base_instance = 0;
                    draw_command.base_vertex = 0;
                    draw_command.cnt = 6;
                    draw_command.first_idx = 0;
                    draw_command.instance_cnt = 1;

                    draw_commands.push_back(draw_command);
                    glyph_data.push_back(glyph);

                } else {
                    // unknown probe type, throw error? do nothing?
                }
            };

            std::visit(visitor, generic_probe);
        }

        auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;
        rt_collection->addRenderTasks(shader, m_billboard_dummy_mesh, draw_commands, glyph_data);
    }

    if (lhs_rtc->version() < m_version) {
        lhs_rtc->setData(rt_collection, m_version);
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
