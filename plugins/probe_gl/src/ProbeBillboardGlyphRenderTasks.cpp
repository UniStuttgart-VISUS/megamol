#include "ProbeBillboardGlyphRenderTasks.h"

#include "ProbeCalls.h"
#include "ProbeEvents.h"
#include "ProbeGlCalls.h"
#include "mesh/MeshCalls.h"
#include "mmcore/EventCall.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/transform.hpp"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"

bool megamol::probe_gl::ProbeBillboardGlyphRenderTasks::create() {

    AbstractGPURenderTaskDataSource::create();

    // Create local copy of transfer function texture (for compatibility with material pipeline)
    glowl::TextureLayout tex_layout;
    tex_layout.width = 1;
    tex_layout.height = 1;
    tex_layout.depth = 1;
    tex_layout.levels = 1;
    // TODO
    tex_layout.format = GL_RGBA;
    tex_layout.type = GL_FLOAT;
    // TODO
    tex_layout.internal_format = GL_RGBA32F;
    tex_layout.int_parameters = {
        {GL_TEXTURE_MIN_FILTER, GL_NEAREST}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR}, {GL_TEXTURE_WRAP_S, GL_CLAMP}};
    this->m_transfer_function = std::make_shared<glowl::Texture2D>("ProbeTransferFunction", tex_layout, nullptr);
    // TODO intialize with value indicating that no transfer function is connected

    return true;
}

void megamol::probe_gl::ProbeBillboardGlyphRenderTasks::release() {}

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::ProbeBillboardGlyphRenderTasks()
    : m_version(0)
    , m_imgui_context(nullptr)
    , m_transfer_function_Slot("GetTransferFunction", "Slot for accessing a transfer function")
    , m_probes_slot("GetProbes", "Slot for accessing a probe collection")
    , m_event_slot("GetProbeEvents", "")
    , m_billboard_dummy_mesh(nullptr)
    , m_billboard_size_slot("BillBoardSize", "Sets the scaling factor of the texture billboards")
    , m_rendering_mode_slot("RenderingMode", "Glyph rendering mode")
    , m_use_interpolation_slot("UseInterpolation", "Interpolate between samples")
    , m_tf_min(0.0f)
    , m_tf_max(1.0f)
    , m_show_glyphs(true) {

    this->m_transfer_function_Slot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->m_transfer_function_Slot);

    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_event_slot.SetCompatibleCall<megamol::core::CallEventDescription>();
    this->MakeSlotAvailable(&this->m_event_slot);

    this->m_billboard_size_slot << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->m_billboard_size_slot);

    this->m_rendering_mode_slot << new megamol::core::param::EnumParam(0);
    this->m_rendering_mode_slot.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Precomputed");
    this->m_rendering_mode_slot.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Realtime");
    this->MakeSlotAvailable(&this->m_rendering_mode_slot);

    this->m_use_interpolation_slot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->m_use_interpolation_slot);
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
        rhs_rtc->setData(rt_collection, 0);
        if (!(*rhs_rtc)(0)) return false;
    }

    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
    if (mtlc == NULL) return false;
    if (!(*mtlc)(0)) return false;

    // create an empty dummy mesh, actual billboard geometry will be build in vertex shader
    if (m_billboard_dummy_mesh == nullptr) {
        std::vector<void*> data_ptrs = {};
        std::vector<size_t> byte_sizes = {};
        std::vector<uint32_t> indices = {0, 1, 2, 3, 4, 5};
        std::vector<glowl::VertexLayout> vertex_layout = {};

        m_billboard_dummy_mesh = std::make_shared<glowl::Mesh>(
            data_ptrs, byte_sizes, indices.data(), 6 * 4, vertex_layout, GL_UNSIGNED_INT, GL_STATIC_DRAW, GL_TRIANGLES);
    }

    probe::CallProbes* pc = this->m_probes_slot.CallAs<probe::CallProbes>();
    if (pc == NULL) return false;
    if (!(*pc)(0)) return false;


    auto* tfc = this->m_transfer_function_Slot.CallAs<core::view::CallGetTransferFunction>();
    if (tfc != NULL) {
        ((*tfc)(0));
    }

    bool something_has_changed = pc->hasUpdate() || mtlc->hasUpdate() || this->m_billboard_size_slot.IsDirty() ||
                                 this->m_rendering_mode_slot.IsDirty() || ((tfc != NULL) ? tfc->IsDirty() : false);

    if (something_has_changed) {
        ++m_version;

        this->m_billboard_size_slot.ResetDirty();
        this->m_rendering_mode_slot.ResetDirty();
        auto gpu_mtl_storage = mtlc->getData();
        auto probes = pc->getData();

        rt_collection->clear();

        auto probe_cnt = probes->getProbeCount();

        m_textured_glyph_data.clear();
        m_vector_probe_glyph_data.clear();
        m_scalar_probe_glyph_data.clear();

        m_textured_gylph_draw_commands.clear();
        m_vector_probe_gylph_draw_commands.clear();
        m_scalar_probe_gylph_draw_commands.clear();

        m_textured_gylph_draw_commands.reserve(probe_cnt);
        m_textured_glyph_data.reserve(probe_cnt);

        m_vector_probe_gylph_draw_commands.reserve(probe_cnt);
        m_vector_probe_glyph_data.reserve(probe_cnt);

        m_scalar_probe_gylph_draw_commands.reserve(probe_cnt);
        m_scalar_probe_glyph_data.reserve(probe_cnt);

        // draw command looks the same for all billboards because geometry is reused
        glowl::DrawElementsCommand draw_command;
        draw_command.base_instance = 0;
        draw_command.base_vertex = 0;
        draw_command.cnt = 6;
        draw_command.first_idx = 0;
        draw_command.instance_cnt = 1;

        // scale in constant over all billboards
        float scale = this->m_billboard_size_slot.Param<core::param::FloatParam>()->Value();

        if (m_rendering_mode_slot.Param<core::param::EnumParam>()->Value() == 0) {
            // use precomputed textures if available

            if (gpu_mtl_storage->getMaterials()[0].textures.size() > 0) {
                for (int probe_idx = 0; probe_idx < probe_cnt; ++probe_idx) {

                    assert(probe_cnt <= (gpu_mtl_storage->getMaterials()[0].textures.size() * 2048));

                    auto generic_probe = probes->getGenericProbe(probe_idx);

                    GLuint64 texture_handle =
                        gpu_mtl_storage->getMaterials()[0].textures[probe_idx / 2048]->getTextureHandle();
                    float slice_idx = probe_idx % 2048;
                    gpu_mtl_storage->getMaterials()[0].textures[probe_idx / 2048]->makeResident();

                    auto visitor = [draw_command, scale, texture_handle, slice_idx, probe_idx, this](auto&& arg) {
                        using T = std::decay_t<decltype(arg)>;

                        auto glyph_data = createTexturedGlyphData(arg, probe_idx, texture_handle, slice_idx, scale);
                        m_textured_gylph_draw_commands.push_back(draw_command);
                        this->m_textured_glyph_data.push_back(glyph_data);
                    };

                    std::visit(visitor, generic_probe);
                }
            }

        } else {
            // Update transfer texture only if it available and has changed
            if (tfc != NULL) {
                if (tfc->IsDirty()) {
                    //++m_version;

                    tfc->ResetDirty();

                    this->m_transfer_function->makeNonResident();
                    this->m_transfer_function.reset();

                    GLenum err = glGetError();
                    if (err != GL_NO_ERROR) {
                        // "Do something cop!"
                        std::cerr << "GL error during transfer function update" << err << std::endl;
                    }

                    glowl::TextureLayout tex_layout;
                    tex_layout.width = tfc->TextureSize();
                    tex_layout.height = 1;
                    tex_layout.depth = 1;
                    tex_layout.levels = 1;
                    // TODO
                    tex_layout.format = GL_RGBA;
                    tex_layout.type = GL_FLOAT;
                    // TODO
                    tex_layout.internal_format = GL_RGBA32F;
                    tex_layout.int_parameters = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST},
                        {GL_TEXTURE_MAG_FILTER, GL_LINEAR}, {GL_TEXTURE_WRAP_S, GL_CLAMP}};
                    this->m_transfer_function = std::make_shared<glowl::Texture2D>(
                        "ProbeTransferFunction", tex_layout, (GLvoid*)tfc->GetTextureData());

                    m_tf_min = std::get<0>(tfc->Range());
                    m_tf_max = std::get<1>(tfc->Range());
                }
            }

            // TODO get transfer function texture from material
            GLuint64 texture_handle = this->m_transfer_function->getTextureHandle();
            this->m_transfer_function->makeResident();

            for (int probe_idx = 0; probe_idx < probe_cnt; ++probe_idx) {

                auto generic_probe = probes->getGenericProbe(probe_idx);

                auto visitor = [draw_command, scale, probe_idx, texture_handle, this](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, probe::FloatProbe>) {

                        auto glyph_data = createScalarProbeGlyphData(arg, probe_idx, scale);
                        glyph_data.tf_texture_handle = texture_handle;
                        m_scalar_probe_gylph_draw_commands.push_back(draw_command);
                        this->m_scalar_probe_glyph_data.push_back(glyph_data);

                    } else if constexpr (std::is_same_v<T, probe::IntProbe>) {
                        // TODO
                    } else if constexpr (std::is_same_v<T, probe::Vec4Probe>) {

                        auto glyph_data = createVectorProbeGlyphData(arg, probe_idx, scale);
                        glyph_data.tf_texture_handle = texture_handle;
                        glyph_data.tf_min = m_tf_min;
                        glyph_data.tf_max = m_tf_max;
                        m_vector_probe_gylph_draw_commands.push_back(draw_command);
                        this->m_vector_probe_glyph_data.push_back(glyph_data);

                    } else {
                        // unknown probe type, throw error? do nothing?
                    }
                };

                std::visit(visitor, generic_probe);
            }

            // scan all scalar probes to compute global min/max
            float min = std::numeric_limits<float>::max();
            float max = std::numeric_limits<float>::min();
            for (auto& data : m_scalar_probe_glyph_data) {
                min = std::min(data.min_value, min);
                max = std::max(data.max_value, max);
            }
            for (auto& data : m_scalar_probe_glyph_data) {
                data.min_value = min;
                data.max_value = max;
            }
        }

        if (!m_textured_gylph_draw_commands.empty()) {
            auto const& textured_shader = gpu_mtl_storage->getMaterials()[0].shader_program;
            rt_collection->addRenderTasks(
                textured_shader, m_billboard_dummy_mesh, m_textured_gylph_draw_commands, m_textured_glyph_data);
        }

        if (!m_scalar_probe_gylph_draw_commands.empty()) {
            auto const& scalar_shader = gpu_mtl_storage->getMaterials()[1].shader_program;
            rt_collection->addRenderTasks(
                scalar_shader, m_billboard_dummy_mesh, m_scalar_probe_gylph_draw_commands, m_scalar_probe_glyph_data);
        }

        if (!m_vector_probe_gylph_draw_commands.empty()) {
            auto const& vector_shader = gpu_mtl_storage->getMaterials()[2].shader_program;
            rt_collection->addRenderTasks(
                vector_shader, m_billboard_dummy_mesh, m_vector_probe_gylph_draw_commands, m_vector_probe_glyph_data);
        }

        std::array<PerFrameData, 1> data;
        data[0].use_interpolation = m_use_interpolation_slot.Param<core::param::BoolParam>()->Value();

        if (rt_collection->getPerFrameBuffers().empty()) {
            rt_collection->addPerFrameDataBuffer(data, 1);
        }
    }

    if (this->m_use_interpolation_slot.IsDirty()) {
        this->m_use_interpolation_slot.ResetDirty();


        std::array<PerFrameData, 1> data;
        data[0].use_interpolation = m_use_interpolation_slot.Param<core::param::BoolParam>()->Value();

        if (!rt_collection->getPerFrameBuffers().empty()) {
            rt_collection->addPerFrameDataBuffer(data, 1);
        } else {
            rt_collection->updatePerFrameDataBuffer(data, 1);
        }
    }

    // check for pending events
    auto call_event_storage = this->m_event_slot.CallAs<core::CallEvent>();
    if (call_event_storage != NULL) {
        if ((!(*call_event_storage)(0))) return false;

        auto event_collection = call_event_storage->getData();

        // process pobe clear selection events
        {
            auto pending_clearselection_events = event_collection->get<ProbeClearSelection>();
            for (auto& evt : pending_clearselection_events) {
                for (auto& draw_data : m_vector_probe_glyph_data) {
                    draw_data.state = 0;
                }

                // TODO this breaks chaining...
                rt_collection->clear();

                if (m_show_glyphs) {
                    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
                    if (mtlc == NULL) return false;

                    auto gpu_mtl_storage = mtlc->getData();

                    if (!m_textured_gylph_draw_commands.empty()) {
                        auto const& textured_shader = gpu_mtl_storage->getMaterials()[0].shader_program;
                        rt_collection->addRenderTasks(textured_shader, m_billboard_dummy_mesh,
                            m_textured_gylph_draw_commands, m_textured_glyph_data);
                    }

                    if (!m_scalar_probe_glyph_data.empty()) {
                        auto const& scalar_shader = gpu_mtl_storage->getMaterials()[1].shader_program;
                        rt_collection->addRenderTasks(scalar_shader, m_billboard_dummy_mesh,
                            m_scalar_probe_gylph_draw_commands, m_scalar_probe_glyph_data);
                    }

                    if (!m_vector_probe_glyph_data.empty()) {
                        auto const& vector_shader = gpu_mtl_storage->getMaterials()[2].shader_program;
                        rt_collection->addRenderTasks(vector_shader, m_billboard_dummy_mesh,
                            m_vector_probe_gylph_draw_commands, m_vector_probe_glyph_data);
                    }
                }
            }
        }

        // process probe highlight events
        {
            auto pending_highlight_events = event_collection->get<ProbeHighlight>();
            for (auto& evt : pending_highlight_events) {
                std::array<GlyphVectorProbeData, 1> per_probe_data = {m_vector_probe_glyph_data[evt.obj_id]};
                per_probe_data[0].state = 1;

                rt_collection->updatePerDrawData(evt.obj_id, per_probe_data);

                bool my_tool_active = true;
                float my_color[4] = {0.0, 0.0, 0.0, 0.0};

                // ImGui::NewFrame();
                // Create a window called "My First Tool", with a menu bar.
                auto ctx = reinterpret_cast<ImGuiContext*>(this->GetCoreInstance()->GetCurrentImGuiContext());
                if (ctx != nullptr) {
                    ImGui::SetCurrentContext(ctx);
                    ImGui::Begin("My First Tool", &my_tool_active, ImGuiWindowFlags_MenuBar);
                    if (ImGui::BeginMenuBar()) {
                        if (ImGui::BeginMenu("File")) {
                            if (ImGui::MenuItem("Open..", "Ctrl+O")) { /* Do stuff */
                            }
                            if (ImGui::MenuItem("Save", "Ctrl+S")) { /* Do stuff */
                            }
                            if (ImGui::MenuItem("Close", "Ctrl+W")) {
                                my_tool_active = false;
                            }
                            ImGui::EndMenu();
                        }
                        ImGui::EndMenuBar();
                    }

                    // Edit a color (stored as ~4 floats)
                    ImGui::ColorEdit4("Color", my_color);

                    // Plot some values
                    const float my_values[] = {0.2f, 0.1f, 1.0f, 0.5f, 0.9f, 2.2f};
                    ImGui::PlotLines("Frame Times", my_values, IM_ARRAYSIZE(my_values));

                    // Display contents in a scrolling region
                    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Important Stuff");
                    ImGui::BeginChild("Scrolling");
                    for (int n = 0; n < 50; n++) ImGui::Text("%04d: Some text", n);
                    ImGui::EndChild();
                    ImGui::End();
                }
            }
        }

        // process probe dehighlight events
        {
            auto pending_dehighlight_events = event_collection->get<ProbeDehighlight>();
            for (auto& evt : pending_dehighlight_events) {
                std::array<GlyphVectorProbeData, 1> per_probe_data = {m_vector_probe_glyph_data[evt.obj_id]};
                rt_collection->updatePerDrawData(evt.obj_id, per_probe_data);
            }
        }

        // process probe selection events
        {
            auto pending_select_events = event_collection->get<ProbeSelect>();
            for (auto& evt : pending_select_events) {
                m_vector_probe_glyph_data[evt.obj_id].state = 2;
                std::array<GlyphVectorProbeData, 1> per_probe_data = {m_vector_probe_glyph_data[evt.obj_id]};

                rt_collection->updatePerDrawData(evt.obj_id, per_probe_data);
            }
        }

        // process probe deselection events
        {
            auto pending_deselect_events = event_collection->get<ProbeDeselect>();
            for (auto& evt : pending_deselect_events) {
                m_vector_probe_glyph_data[evt.obj_id].state = 0;
                std::array<GlyphVectorProbeData, 1> per_probe_data = {m_vector_probe_glyph_data[evt.obj_id]};

                rt_collection->updatePerDrawData(evt.obj_id, per_probe_data);
            }
        }

        // process probe exclusive selection events
        {
            auto pending_selectExclusive_events = event_collection->get<ProbeSelectExclusive>();
            if (!pending_selectExclusive_events.empty()) {
                for (auto& draw_data : m_vector_probe_glyph_data) {
                    draw_data.state = 0;
                }
                // multiple exclusive selections make no sense, just applay the last one
                m_vector_probe_glyph_data[pending_selectExclusive_events.back().obj_id].state = 2;

                // TODO this breaks chaining...
                rt_collection->clear();

                if (m_show_glyphs) {
                    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
                    if (mtlc == NULL) return false;

                    auto gpu_mtl_storage = mtlc->getData();

                    if (!m_textured_gylph_draw_commands.empty()) {
                        auto const& textured_shader = gpu_mtl_storage->getMaterials()[0].shader_program;
                        rt_collection->addRenderTasks(textured_shader, m_billboard_dummy_mesh,
                            m_textured_gylph_draw_commands, m_textured_glyph_data);
                    }

                    if (!m_scalar_probe_glyph_data.empty()) {
                        auto const& scalar_shader = gpu_mtl_storage->getMaterials()[1].shader_program;
                        rt_collection->addRenderTasks(scalar_shader, m_billboard_dummy_mesh,
                            m_scalar_probe_gylph_draw_commands, m_scalar_probe_glyph_data);
                    }

                    if (!m_vector_probe_glyph_data.empty()) {
                        auto const& vector_shader = gpu_mtl_storage->getMaterials()[2].shader_program;
                        rt_collection->addRenderTasks(vector_shader, m_billboard_dummy_mesh,
                            m_vector_probe_gylph_draw_commands, m_vector_probe_glyph_data);
                    }
                }
            }
        }

        // process probe selection toggle events
        {
            auto pending_select_events = event_collection->get<ProbeSelectToggle>();
            for (auto& evt : pending_select_events) {
                m_vector_probe_glyph_data[evt.obj_id].state = m_vector_probe_glyph_data[evt.obj_id].state == 2 ? 0 : 2;
                std::array<GlyphVectorProbeData, 1> per_probe_data = {m_vector_probe_glyph_data[evt.obj_id]};

                rt_collection->updatePerDrawData(evt.obj_id, per_probe_data);
            }
        }

        // process toggle show glyph events
        {
            auto pending_deselect_events = event_collection->get<ToggleShowGlyphs>();
            for (auto& evt : pending_deselect_events) {
                m_show_glyphs = !m_show_glyphs;

                if (m_show_glyphs) {
                    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
                    if (mtlc == NULL) return false;

                    auto gpu_mtl_storage = mtlc->getData();

                    if (!m_textured_gylph_draw_commands.empty()) {
                        auto const& textured_shader = gpu_mtl_storage->getMaterials()[0].shader_program;
                        rt_collection->addRenderTasks(textured_shader, m_billboard_dummy_mesh,
                            m_textured_gylph_draw_commands, m_textured_glyph_data);
                    }

                    if (!m_scalar_probe_glyph_data.empty()) {
                        auto const& scalar_shader = gpu_mtl_storage->getMaterials()[1].shader_program;
                        rt_collection->addRenderTasks(scalar_shader, m_billboard_dummy_mesh,
                            m_scalar_probe_gylph_draw_commands, m_scalar_probe_glyph_data);
                    }

                    if (!m_vector_probe_glyph_data.empty()) {
                        auto const& vector_shader = gpu_mtl_storage->getMaterials()[2].shader_program;
                        rt_collection->addRenderTasks(vector_shader, m_billboard_dummy_mesh,
                            m_vector_probe_gylph_draw_commands, m_vector_probe_glyph_data);
                    }
                } else {
                    // TODO this breaks chaining...
                    rt_collection->clear();
                }
            }
        }
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

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::GlyphScalarProbeData
megamol::probe_gl::ProbeBillboardGlyphRenderTasks::createScalarProbeGlyphData(
    probe::FloatProbe const& probe, int probe_id, float scale) {
    GlyphScalarProbeData glyph_data;
    glyph_data.position = glm::vec4(probe.m_position[0] + probe.m_direction[0] * (probe.m_begin * 1.25f),
        probe.m_position[1] + probe.m_direction[1] * (probe.m_begin * 1.25f),
        probe.m_position[2] + probe.m_direction[2] * (probe.m_begin * 1.25f), 1.0f);

    glyph_data.probe_direction = glm::vec4(probe.m_direction[0], probe.m_direction[1], probe.m_direction[2], 1.0f);

    glyph_data.scale = scale;

    if (probe.getSamplingResult()->samples.size() > 32) {
        // TODO print warning/error message
    }

    glyph_data.min_value = probe.getSamplingResult()->min_value;
    glyph_data.max_value = probe.getSamplingResult()->max_value;

    glyph_data.sample_cnt = std::min(static_cast<size_t>(32), probe.getSamplingResult()->samples.size());

    for (int i = 0; i < glyph_data.sample_cnt; ++i) {
        glyph_data.samples[i] = probe.getSamplingResult()->samples[i];
    }

    glyph_data.probe_id = probe_id;

    return glyph_data;
}

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::GlyphVectorProbeData
megamol::probe_gl::ProbeBillboardGlyphRenderTasks::createVectorProbeGlyphData(
    probe::Vec4Probe const& probe, int probe_id, float scale) {

    GlyphVectorProbeData glyph_data;
    glyph_data.position = glm::vec4(probe.m_position[0] + probe.m_direction[0] * (probe.m_begin * 1.25f),
        probe.m_position[1] + probe.m_direction[1] * (probe.m_begin * 1.25f),
        probe.m_position[2] + probe.m_direction[2] * (probe.m_begin * 1.25f), 1.0f);

    glyph_data.probe_direction = glm::vec4(probe.m_direction[0], probe.m_direction[1], probe.m_direction[2], 1.0f);

    glyph_data.scale = scale;

    if (probe.getSamplingResult()->samples.size() > 32) {
        // TODO print warning/error message
    }

    glyph_data.sample_cnt = std::min(static_cast<size_t>(32), probe.getSamplingResult()->samples.size());

    for (int i = 0; i < glyph_data.sample_cnt; ++i) {
        glyph_data.samples[i] = probe.getSamplingResult()->samples[i];
    }

    glyph_data.probe_id = probe_id;

    return glyph_data;
}
