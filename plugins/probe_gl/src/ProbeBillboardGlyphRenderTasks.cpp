#include "ProbeBillboardGlyphRenderTasks.h"

#include "ProbeEvents.h"
#include "ProbeGlCalls.h"
#include "mesh_gl/MeshCalls_gl.h"
#include "mmstd/event/EventCall.h"
#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"
#include "probe/ProbeCalls.h"

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/transform.hpp"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"
#include <imgui_internal.h>

bool megamol::probe_gl::ProbeBillboardGlyphRenderTasks::create() {

    mesh_gl::AbstractGPURenderTaskDataSource::create();

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
    try {
        this->m_transfer_function = std::make_shared<glowl::Texture2D>("ProbeTransferFunction", tex_layout, nullptr);
    } catch (glowl::TextureException const& exc) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error on transfer texture view pre-creation: %s. [%s, %s, line %d]\n", exc.what(), __FILE__, __FUNCTION__,
            __LINE__);
    }
    // TODO intialize with value indicating that no transfer function is connected
    this->m_transfer_function->makeResident();

    return true;
}

void megamol::probe_gl::ProbeBillboardGlyphRenderTasks::release() {}

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::ProbeBillboardGlyphRenderTasks()
        : m_version(0)
        , m_imgui_context(nullptr)
        , m_transfer_function_Slot("GetTransferFunction", "Slot for accessing a transfer function")
        , m_probes_slot("GetProbes", "Slot for accessing a probe collection")
        , m_event_slot("GetProbeEvents", "")
        , m_material_slot("GetProbeGlyphMaterials", "")
        , m_billboard_dummy_mesh(nullptr)
        , m_billboard_size_slot("BillBoardSize", "Sets the scaling factor of the texture billboards")
        , m_rendering_mode_slot("RenderingMode", "Glyph rendering mode")
        , m_use_interpolation_slot("UseInterpolation", "Interpolate between samples")
        , m_show_canvas_slot("ShowGlyphCanvas", "Render glyphs with opaque background")
        , m_canvas_color_slot("GlyphCanvasColor", "Color used for the background of individual glyphs")
        , m_tf_range({0.0f, 0.0f})
        , m_show_glyphs(true) {

    this->m_transfer_function_Slot.SetCompatibleCall<mmstd_gl::CallGetTransferFunctionGLDescription>();
    this->MakeSlotAvailable(&this->m_transfer_function_Slot);

    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_material_slot.SetCompatibleCall<mesh_gl::CallGPUMaterialDataDescription>();
    this->MakeSlotAvailable(&this->m_material_slot);

    this->m_event_slot.SetCompatibleCall<megamol::core::CallEventDescription>();
    this->MakeSlotAvailable(&this->m_event_slot);

    this->m_billboard_size_slot << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->m_billboard_size_slot);

    this->m_rendering_mode_slot << new megamol::core::param::EnumParam(0);
    this->m_rendering_mode_slot.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Precomputed");
    this->m_rendering_mode_slot.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Realtime");
    this->m_rendering_mode_slot.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "ClusterID");
    this->MakeSlotAvailable(&this->m_rendering_mode_slot);

    this->m_use_interpolation_slot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->m_use_interpolation_slot);

    this->m_show_canvas_slot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->m_show_canvas_slot);

    this->m_canvas_color_slot << new core::param::ColorParam(1.0, 1.0, 1.0, 1.0);
    this->MakeSlotAvailable(&this->m_canvas_color_slot);
}

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::~ProbeBillboardGlyphRenderTasks() {}

bool megamol::probe_gl::ProbeBillboardGlyphRenderTasks::getDataCallback(core::Call& caller) {

    auto err = glGetError();
    if (err != GL_NO_ERROR) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unexpeced OpenGL error: %i. [%s, %s, line %d]\n", err, __FILE__, __FUNCTION__, __LINE__);
    }

    mesh_gl::CallGPURenderTaskData* lhs_rtc = dynamic_cast<mesh_gl::CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL) {
        return false;
    }

    mesh_gl::CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<mesh_gl::CallGPURenderTaskData>();

    std::vector<std::shared_ptr<mesh_gl::GPURenderTaskCollection>> gpu_render_tasks;
    if (rhs_rtc != nullptr) {
        if (!(*rhs_rtc)(0)) {
            return false;
        }
        if (rhs_rtc->hasUpdate()) {
            ++m_version;
        }
        gpu_render_tasks = rhs_rtc->getData();
    }
    gpu_render_tasks.push_back(m_rendertask_collection.first);

    mesh_gl::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh_gl::CallGPUMaterialData>();
    if (mtlc == NULL)
        return false;
    if (!(*mtlc)(0))
        return false;

    // create an empty dummy mesh, actual billboard geometry will be build in vertex shader
    if (m_billboard_dummy_mesh == nullptr) {
        std::vector<void const*> data_ptrs = {};
        std::vector<size_t> byte_sizes = {};
        std::vector<uint32_t> indices = {0, 1, 2, 3, 4, 5};
        std::vector<glowl::VertexLayout> vertex_layout = {};

        m_billboard_dummy_mesh = std::make_shared<glowl::Mesh>(
            data_ptrs, byte_sizes, vertex_layout, indices.data(), 6 * 4, GL_UNSIGNED_INT, GL_TRIANGLES, GL_STATIC_DRAW);
    }

    probe::CallProbes* pc = this->m_probes_slot.CallAs<probe::CallProbes>();
    if (pc == NULL) {
        return false;
    }
    if (!(*pc)(0)) {
        return false;
    }

    auto* tfc = this->m_transfer_function_Slot.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
    if (tfc != NULL) {
        ((*tfc)(0));
    }

    bool something_has_changed = pc->hasUpdate() || mtlc->hasUpdate() || this->m_billboard_size_slot.IsDirty() ||
                                 this->m_rendering_mode_slot.IsDirty();

    if (something_has_changed) {
        ++m_version;

        this->m_rendering_mode_slot.ResetDirty();
        auto gpu_mtl_storage = mtlc->getData();
        auto probes = pc->getData();

        auto visitor = [this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, probe::FloatProbe>) {

            } else if constexpr (std::is_same_v<T, std::array<float, 2>>) {
                m_tf_range = arg;
            } else if constexpr (std::is_same_v<T, std::array<int, 2>>) {
                // TODO
            } else {
                // unknown probe type, throw error? do nothing?
            }
        };

        std::visit(visitor, probes->getGenericGlobalMinMax());

        //for (auto& identifier : m_rendertask_collection.second) {
        //    m_rendertask_collection.first->deleteRenderTask(identifier);
        //}
        m_rendertask_collection.first->clear();
        m_rendertask_collection.second.clear();


        auto probe_cnt = probes->getProbeCount();

        m_type_index_map.clear();
        m_type_index_map.reserve(probe_cnt);

        m_textured_glyph_identifiers.clear();
        m_vector_probe_glyph_identifiers.clear();
        m_scalar_probe_glyph_identifiers.clear();
        m_scalar_distribution_probe_glyph_identifiers.clear();
        m_clusterID_glyph_identifiers.clear();

        m_textured_glyph_data.clear();
        m_vector_probe_glyph_data.clear();
        m_scalar_probe_glyph_data.clear();
        m_scalar_distribution_probe_glyph_data.clear();
        m_clusterID_glyph_data.clear();

        m_textured_gylph_draw_commands.clear();
        m_vector_probe_gylph_draw_commands.clear();
        m_scalar_probe_gylph_draw_commands.clear();
        m_scalar_distribution_probe_gylph_draw_commands.clear();
        m_clusterID_gylph_draw_commands.clear();

        m_textured_gylph_draw_commands.reserve(probe_cnt);
        m_textured_glyph_data.reserve(probe_cnt);

        m_vector_probe_gylph_draw_commands.reserve(probe_cnt);
        m_vector_probe_glyph_data.reserve(probe_cnt);

        m_scalar_probe_gylph_draw_commands.reserve(probe_cnt);
        m_scalar_probe_glyph_data.reserve(probe_cnt);

        m_scalar_distribution_probe_gylph_draw_commands.reserve(probe_cnt);
        m_scalar_distribution_probe_glyph_data.reserve(probe_cnt);

        m_clusterID_gylph_draw_commands.reserve(probe_cnt);
        m_clusterID_glyph_data.reserve(probe_cnt);

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

            mesh_gl::GPUMaterialCollection::Material mat;
            for (int i = 0; i < gpu_mtl_storage.size(); ++i) {
                auto query = gpu_mtl_storage[i]->getMaterial("ProbeBillboard_Textured");
                if (mat.shader_program != nullptr) {
                    mat = query;
                    break;
                }
            }

            if (mat.textures.size() > 0) {
                for (int probe_idx = 0; probe_idx < probe_cnt; ++probe_idx) {

                    assert(probe_cnt <=
                           (gpu_mtl_storage.front()->getMaterial("ProbeBillboard_Textured").textures.size() * 2048));

                    auto generic_probe = probes->getGenericProbe(probe_idx);

                    GLuint64 texture_handle = mat.textures[probe_idx / 2048]->getTextureHandle();
                    float slice_idx = probe_idx % 2048;
                    mat.textures[probe_idx / 2048]->makeResident();

                    auto visitor = [draw_command, scale, texture_handle, slice_idx, probe_idx, this](auto&& arg) {
                        using T = std::decay_t<decltype(arg)>;

                        auto glyph_data = createTexturedGlyphData(arg, probe_idx, texture_handle, slice_idx, scale);
                        m_textured_gylph_draw_commands.push_back(draw_command);
                        this->m_textured_glyph_data.push_back(glyph_data);

                        m_textured_glyph_identifiers.emplace_back(
                            std::string(FullName()) + "_tg_" + std::to_string(probe_idx));

                        this->m_type_index_map.push_back({std::type_index(typeid(TexturedGlyphData)), probe_idx});
                    };

                    std::visit(visitor, generic_probe);
                }
            }

        } else if (m_rendering_mode_slot.Param<core::param::EnumParam>()->Value() == 1) {

            for (int probe_idx = 0; probe_idx < probe_cnt; ++probe_idx) {

                auto generic_probe = probes->getGenericProbe(probe_idx);

                auto visitor = [draw_command, scale, probe_idx, this](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, probe::FloatProbe>) {

                        auto sp_idx = m_scalar_probe_glyph_data.size();

                        auto glyph_data = createScalarProbeGlyphData(arg, probe_idx, scale);
                        m_scalar_probe_gylph_draw_commands.push_back(draw_command);
                        this->m_scalar_probe_glyph_data.push_back(glyph_data);

                        m_scalar_probe_glyph_identifiers.emplace_back(
                            std::string(FullName()) + "_sg_" + std::to_string(probe_idx));

                        this->m_type_index_map.push_back({std::type_index(typeid(GlyphScalarProbeData)), sp_idx});

                    } else if constexpr (std::is_same_v<T, probe::FloatDistributionProbe>) {
                        auto sp_idx = m_scalar_distribution_probe_glyph_data.size();

                        auto glyph_data = createScalarDistributionProbeGlyphData(arg, probe_idx, scale);
                        m_scalar_distribution_probe_gylph_draw_commands.push_back(draw_command);
                        this->m_scalar_distribution_probe_glyph_data.push_back(glyph_data);

                        m_scalar_distribution_probe_glyph_identifiers.emplace_back(
                            std::string(FullName()) + "_sdg_" + std::to_string(probe_idx));

                        this->m_type_index_map.push_back(
                            {std::type_index(typeid(GlyphScalarDistributionProbeData)), sp_idx});
                    } else if constexpr (std::is_same_v<T, probe::IntProbe>) {
                        // TODO
                    } else if constexpr (std::is_same_v<T, probe::Vec4Probe>) {

                        auto vp_idx = m_vector_probe_glyph_data.size();

                        auto glyph_data = createVectorProbeGlyphData(arg, probe_idx, scale);
                        m_vector_probe_gylph_draw_commands.push_back(draw_command);
                        this->m_vector_probe_glyph_data.push_back(glyph_data);

                        m_vector_probe_glyph_identifiers.emplace_back(
                            std::string(FullName()) + "_vg_" + std::to_string(probe_idx));

                        this->m_type_index_map.push_back({std::type_index(typeid(GlyphVectorProbeData)), vp_idx});

                    } else {
                        // unknown probe type, throw error? do nothing?
                    }
                };

                std::visit(visitor, generic_probe);
            }

            // scan all scalar probes to compute global min/max
            //float min = std::numeric_limits<float>::max();
            //float max = std::numeric_limits<float>::min();
            //for (auto& data : m_scalar_probe_glyph_data) {
            //    min = std::min(data.min_value, min);
            //    max = std::max(data.max_value, max);
            //}
            //for (auto& data : m_scalar_probe_glyph_data) {
            //    data.min_value = min;
            //    data.max_value = max;
            //}
        } else {
            int total_cluster_cnt = 0;
            for (int probe_idx = 0; probe_idx < probe_cnt; ++probe_idx) {

                auto generic_probe = probes->getGenericProbe(probe_idx);

                auto visitor = [draw_command, scale, probe_idx, &total_cluster_cnt, this](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;

                    auto glyph_data = createClusterIDGlyphData(arg, probe_idx, scale);
                    m_clusterID_gylph_draw_commands.push_back(draw_command);
                    this->m_clusterID_glyph_data.push_back(glyph_data);

                    total_cluster_cnt = std::max(total_cluster_cnt, glyph_data.cluster_id);

                    m_clusterID_glyph_identifiers.push_back(
                        std::string(FullName()) + "_cg_" + std::to_string(probe_idx));

                    this->m_type_index_map.push_back({std::type_index(typeid(GlyphClusterIDData)), probe_idx});
                };

                std::visit(visitor, generic_probe);
            }

            for (auto& glyph_data : m_clusterID_glyph_data) {
                glyph_data.total_cluster_cnt = total_cluster_cnt;
            }
        }

        addAllRenderTasks();
    }

    bool per_frame_data_has_changed = this->m_use_interpolation_slot.IsDirty() || this->m_show_canvas_slot.IsDirty() ||
                                      this->m_canvas_color_slot.IsDirty() || ((tfc != NULL) ? tfc->IsDirty() : false) ||
                                      m_rendertask_collection.first->getPerFrameBuffers().empty();

    if (per_frame_data_has_changed) {
        this->m_use_interpolation_slot.ResetDirty();
        this->m_show_canvas_slot.ResetDirty();
        this->m_canvas_color_slot.ResetDirty();

        if (tfc != NULL) {
            tfc->SetRange(m_tf_range);
            ((*tfc)());
            m_tf_range = tfc->Range();
        }

        // Update transfer texture only if it available and has changed
        if (tfc != NULL) {

            if (tfc->IsDirty()) {
                //++m_version;
                tfc->ResetDirty();

                this->m_transfer_function->makeNonResident();
                this->m_transfer_function.reset();

                {
                    GLenum err = glGetError();
                    if (err != GL_NO_ERROR) {
                        // "Do something cop!"
                        std::cerr << "GL error during transfer function update" << err << std::endl;
                    }
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
                tex_layout.int_parameters = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR},
                    {GL_TEXTURE_WRAP_S, GL_CLAMP}};
                try {
                    this->m_transfer_function = std::make_shared<glowl::Texture2D>(
                        "ProbeTransferFunction", tex_layout, (GLvoid*)tfc->GetTextureData());
                } catch (glowl::TextureException const& exc) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Error on transfer texture view creation: %s. [%s, %s, line %d]\n", exc.what(), __FILE__,
                        __FUNCTION__, __LINE__);
                }

                this->m_transfer_function->makeResident();
                {
                    auto err = glGetError();
                    if (err != GL_NO_ERROR) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "Error on making transfer texture view resident: %i. [%s, %s, line %d]\n", err, __FILE__,
                            __FUNCTION__, __LINE__);
                    }
                }

                m_tf_range = tfc->Range();
            }
        }

        GLuint64 texture_handle = this->m_transfer_function->getTextureHandle();

        std::array<PerFrameData, 1> data;
        data[0].use_interpolation = m_use_interpolation_slot.Param<core::param::BoolParam>()->Value();
        data[0].show_canvas = m_show_canvas_slot.Param<core::param::BoolParam>()->Value();
        data[0].canvas_color = m_canvas_color_slot.Param<core::param::ColorParam>()->Value();
        data[0].tf_texture_handle = texture_handle;
        data[0].tf_min = std::get<0>(m_tf_range);
        data[0].tf_max = std::get<1>(m_tf_range);

        std::string identifier = std::string(FullName()) + "_perFrameData";
        if (m_rendertask_collection.first->getPerFrameBuffers().empty()) {
            m_rendertask_collection.first->addPerFrameDataBuffer(identifier, data, 1);
        } else {
            m_rendertask_collection.first->updatePerFrameDataBuffer(identifier, data, 1);
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
            auto pending_events = event_collection->get<ProbeClearSelection>();
            if (!pending_events.empty()) {
                for (auto& draw_data : m_scalar_probe_glyph_data) {
                    draw_data.state = 0;
                }
                for (auto& draw_data : m_vector_probe_glyph_data) {
                    draw_data.state = 0;
                }
                for (auto& draw_data : m_clusterID_glyph_data) {
                    draw_data.state = 0;
                }

                updateAllRenderTasks();
            }
        }

        // process probe highlight events
        {
            auto pending_highlight_events = event_collection->get<ProbeHighlight>();
            for (auto& evt : pending_highlight_events) {
                auto probe_type = m_type_index_map[evt.obj_id].first;
                auto probe_idx = m_type_index_map[evt.obj_id].second;

                if (probe_type == std::type_index(typeid(GlyphScalarProbeData))) {
                    std::array<GlyphScalarProbeData, 1> per_probe_data = {m_scalar_probe_glyph_data[probe_idx]};
                    per_probe_data[0].state = 1;
                    std::string identifier = std::string(FullName()) + "_sg_" + std::to_string(probe_idx);
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                } else if (probe_type == std::type_index(typeid(GlyphVectorProbeData))) {
                    std::array<GlyphVectorProbeData, 1> per_probe_data = {m_vector_probe_glyph_data[probe_idx]};
                    per_probe_data[0].state = 1;
                    std::string identifier = std::string(FullName()) + "_vg_" + std::to_string(probe_idx);
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                } else if (probe_type == std::type_index(typeid(GlyphClusterIDData))) {
                    std::array<GlyphClusterIDData, 1> per_probe_data = {m_clusterID_glyph_data[probe_idx]};
                    per_probe_data[0].state = 1;
                    std::string identifier = std::string(FullName()) + "_cg_" + std::to_string(probe_idx);
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                }

                //  bool my_tool_active = true;
                //  float my_color[4] = {0.0, 0.0, 0.0, 0.0};
                //
                //  // ImGui::NewFrame();
                //  // Create a window called "My First Tool", with a menu bar.
                //  auto ctx = reinterpret_cast<ImGuiContext*>(this->GetCoreInstance()->GetCurrentImGuiContext());
                //  if (ctx != nullptr) {
                //      ImGui::SetCurrentContext(ctx);
                //      ImGui::Begin("My First Tool", &my_tool_active, ImGuiWindowFlags_MenuBar);
                //      if (ImGui::BeginMenuBar()) {
                //          if (ImGui::BeginMenu("File")) {
                //              if (ImGui::MenuItem("Open..", "Ctrl+O")) { /* Do stuff */
                //              }
                //              if (ImGui::MenuItem("Save", "Ctrl+S")) { /* Do stuff */
                //              }
                //              if (ImGui::MenuItem("Close", "Ctrl+W")) {
                //                  my_tool_active = false;
                //              }
                //              ImGui::EndMenu();
                //          }
                //          ImGui::EndMenuBar();
                //      }
                //
                //      // Edit a color (stored as ~4 floats)
                //      ImGui::ColorEdit4("Color", my_color);
                //
                //      // Plot some values
                //      const float my_values[] = {0.2f, 0.1f, 1.0f, 0.5f, 0.9f, 2.2f};
                //      ImGui::PlotLines("Frame Times", my_values, IM_ARRAYSIZE(my_values));
                //
                //      // Display contents in a scrolling region
                //      ImGui::TextColored(ImVec4(1, 1, 0, 1), "Important Stuff");
                //      ImGui::BeginChild("Scrolling");
                //      for (int n = 0; n < 50; n++) ImGui::Text("%04d: Some text", n);
                //      ImGui::EndChild();
                //      ImGui::End();
                //  }
            }
        }

        // process probe dehighlight events
        {
            auto pending_dehighlight_events = event_collection->get<ProbeDehighlight>();
            for (auto& evt : pending_dehighlight_events) {
                auto probe_type = m_type_index_map[evt.obj_id].first;
                auto probe_idx = m_type_index_map[evt.obj_id].second;

                if (probe_type == std::type_index(typeid(GlyphScalarProbeData))) {
                    std::array<GlyphScalarProbeData, 1> per_probe_data = {m_scalar_probe_glyph_data[probe_idx]};
                    std::string identifier = std::string(FullName()) + "_sg_" + std::to_string(probe_idx);
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                } else if (probe_type == std::type_index(typeid(GlyphVectorProbeData))) {
                    std::array<GlyphVectorProbeData, 1> per_probe_data = {m_vector_probe_glyph_data[probe_idx]};
                    std::string identifier = std::string(FullName()) + "_vg_" + std::to_string(probe_idx);
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                } else if (probe_type == std::type_index(typeid(GlyphClusterIDData))) {
                    std::array<GlyphClusterIDData, 1> per_probe_data = {m_clusterID_glyph_data[probe_idx]};
                    std::string identifier = std::string(FullName()) + "_cg_" + std::to_string(probe_idx);
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                }
            }
        }

        // process probe selection events
        {
            auto pending_select_events = event_collection->get<ProbeSelect>();
            for (auto& evt : pending_select_events) {
                auto probe_type = m_type_index_map[evt.obj_id].first;
                auto probe_idx = m_type_index_map[evt.obj_id].second;

                if (probe_type == std::type_index(typeid(GlyphScalarProbeData))) {
                    m_scalar_probe_glyph_data[probe_idx].state = 2;
                    std::array<GlyphScalarProbeData, 1> per_probe_data = {m_scalar_probe_glyph_data[probe_idx]};
                    std::string identifier = std::string(FullName()) + "_ProbeBillboard_Scalar";
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                } else if (probe_type == std::type_index(typeid(GlyphVectorProbeData))) {
                    m_vector_probe_glyph_data[probe_idx].state = 2;
                    std::array<GlyphVectorProbeData, 1> per_probe_data = {m_vector_probe_glyph_data[probe_idx]};
                    std::string identifier = std::string(FullName()) + "_ProbeBillboard_Vector";
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                } else if (probe_type == std::type_index(typeid(GlyphClusterIDData))) {
                    m_clusterID_glyph_data[probe_idx].state = 2;
                    std::array<GlyphClusterIDData, 1> per_probe_data = {m_clusterID_glyph_data[probe_idx]};
                    std::string identifier = std::string(FullName()) + "_ProbeBillboard_ClusterID";
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                }
            }
        }

        // process probe deselection events
        {
            auto pending_deselect_events = event_collection->get<ProbeDeselect>();
            for (auto& evt : pending_deselect_events) {
                auto probe_type = m_type_index_map[evt.obj_id].first;
                auto probe_idx = m_type_index_map[evt.obj_id].second;

                if (probe_type == std::type_index(typeid(GlyphScalarProbeData))) {
                    m_scalar_probe_glyph_data[probe_idx].state = 0;
                    std::array<GlyphScalarProbeData, 1> per_probe_data = {m_scalar_probe_glyph_data[probe_idx]};
                    std::string identifier = std::string(FullName()) + "_sg_" + std::to_string(probe_idx);
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                } else if (probe_type == std::type_index(typeid(GlyphVectorProbeData))) {
                    m_vector_probe_glyph_data[probe_idx].state = 0;
                    std::array<GlyphVectorProbeData, 1> per_probe_data = {m_vector_probe_glyph_data[probe_idx]};
                    std::string identifier = std::string(FullName()) + "_vg_" + std::to_string(probe_idx);
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                } else if (probe_type == std::type_index(typeid(GlyphClusterIDData))) {
                    m_clusterID_glyph_data[probe_idx].state = 0;
                    std::array<GlyphClusterIDData, 1> per_probe_data = {m_clusterID_glyph_data[probe_idx]};
                    std::string identifier = std::string(FullName()) + "_cg_" + std::to_string(probe_idx);
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                }
            }
        }

        // process probe exclusive selection events
        {
            auto pending_events = event_collection->get<ProbeSelectExclusive>();
            if (!pending_events.empty()) {

                for (auto& draw_data : m_scalar_probe_glyph_data) {
                    draw_data.state = 0;
                }
                for (auto& draw_data : m_vector_probe_glyph_data) {
                    draw_data.state = 0;
                }
                for (auto& draw_data : m_clusterID_glyph_data) {
                    draw_data.state = 0;
                }

                auto probe_type = m_type_index_map[pending_events.back().obj_id].first;
                auto probe_idx = m_type_index_map[pending_events.back().obj_id].second;

                // multiple exclusive selections make no sense, just apply the last one
                if (probe_type == std::type_index(typeid(GlyphScalarProbeData))) {
                    m_scalar_probe_glyph_data[probe_idx].state = 2;
                } else if (probe_type == std::type_index(typeid(GlyphVectorProbeData))) {
                    m_vector_probe_glyph_data[probe_idx].state = 2;
                } else if (probe_type == std::type_index(typeid(GlyphClusterIDData))) {
                    m_clusterID_glyph_data[probe_idx].state = 2;
                }

                updateAllRenderTasks();
            }
        }

        // process probe selection toggle events
        {
            auto pending_select_events = event_collection->get<ProbeSelectToggle>();
            for (auto& evt : pending_select_events) {
                auto probe_type = m_type_index_map[evt.obj_id].first;
                auto probe_idx = m_type_index_map[evt.obj_id].second;

                if (probe_type == std::type_index(typeid(GlyphScalarProbeData))) {
                    m_scalar_probe_glyph_data[probe_idx].state =
                        m_scalar_probe_glyph_data[probe_idx].state == 2 ? 0 : 2;
                    std::array<GlyphScalarProbeData, 1> per_probe_data = {m_scalar_probe_glyph_data[probe_idx]};
                    std::string identifier = std::string(FullName()) + "_sg_" + std::to_string(probe_idx);
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                } else if (probe_type == std::type_index(typeid(GlyphVectorProbeData))) {
                    m_vector_probe_glyph_data[probe_idx].state =
                        m_vector_probe_glyph_data[probe_idx].state == 2 ? 0 : 2;
                    std::array<GlyphVectorProbeData, 1> per_probe_data = {m_vector_probe_glyph_data[probe_idx]};
                    std::string identifier = std::string(FullName()) + "_vg_" + std::to_string(probe_idx);
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                } else if (probe_type == std::type_index(typeid(GlyphClusterIDData))) {
                    m_clusterID_glyph_data[probe_idx].state = m_clusterID_glyph_data[probe_idx].state == 2 ? 0 : 2;
                    std::array<GlyphClusterIDData, 1> per_probe_data = {m_clusterID_glyph_data[probe_idx]};
                    std::string identifier = std::string(FullName()) + "_cg_" + std::to_string(probe_idx);
                    m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
                }
            }
        }

        // process toggle show glyph events
        {
            auto pending_deselect_events = event_collection->get<ToggleShowGlyphs>();
            for (auto& evt : pending_deselect_events) {
                m_show_glyphs = !m_show_glyphs;

                if (m_show_glyphs) {
                    addAllRenderTasks();
                } else {
                    clearAllRenderTasks();
                }
            }
        }
    }

    lhs_rtc->setData(gpu_render_tasks, m_version);

    return true;
}

bool megamol::probe_gl::ProbeBillboardGlyphRenderTasks::getMetaDataCallback(core::Call& caller) {

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

bool megamol::probe_gl::ProbeBillboardGlyphRenderTasks::addAllRenderTasks() {
    mesh_gl::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh_gl::CallGPUMaterialData>();
    if (mtlc == NULL)
        return false;
    if (!(*mtlc)(0))
        return false;

    std::shared_ptr<glowl::GLSLProgram> textured_shader(nullptr);
    std::shared_ptr<glowl::GLSLProgram> scalar_shader(nullptr);
    std::shared_ptr<glowl::GLSLProgram> scalar_distribution_shader(nullptr);
    std::shared_ptr<glowl::GLSLProgram> vector_shader(nullptr);
    std::shared_ptr<glowl::GLSLProgram> clusterID_shader(nullptr);

    auto gpu_mtl_storage = mtlc->getData();
    for (int i = 0; i < gpu_mtl_storage.size(); ++i) {
        auto textured_query = gpu_mtl_storage[i]->getMaterials().find("ProbeBillboard_Textured");
        auto scalar_query = gpu_mtl_storage[i]->getMaterials().find("ProbeBillboard_Scalar");
        auto scalar_distribution_query = gpu_mtl_storage[i]->getMaterials().find("ProbeBillboard_ScalarDistribution");
        auto vector_query = gpu_mtl_storage[i]->getMaterials().find("ProbeBillboard_Vector");
        auto clusterID_query = gpu_mtl_storage[i]->getMaterials().find("ProbeBillboard_ClusterID");

        if (textured_query != gpu_mtl_storage[i]->getMaterials().end()) {
            textured_shader = textured_query->second.shader_program;
        }
        if (scalar_query != gpu_mtl_storage[i]->getMaterials().end()) {
            scalar_shader = scalar_query->second.shader_program;
        }
        if (scalar_distribution_query != gpu_mtl_storage[i]->getMaterials().end()) {
            scalar_distribution_shader = scalar_distribution_query->second.shader_program;
        }
        if (vector_query != gpu_mtl_storage[i]->getMaterials().end()) {
            vector_shader = vector_query->second.shader_program;
        }
        if (clusterID_query != gpu_mtl_storage[i]->getMaterials().end()) {
            clusterID_shader = clusterID_query->second.shader_program;
        }
    }

    if (textured_shader == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Could not get ProbeBillboard_Textured material, identifier not found. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
    }
    if (scalar_shader == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Could not get ProbeBillboard_Scalar material, identifier not found. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
    }
    if (scalar_distribution_shader == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Could not get ProbeBillboard_Scalar_Distribution material, identifier not found. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
    }
    if (vector_shader == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Could not get ProbeBillboard_Vector material, identifier not found. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
    }
    if (clusterID_shader == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Could not get ProbeBillboard_ClusterID material, identifier not found. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
    }

    if (!m_textured_gylph_draw_commands.empty()) {
        m_rendertask_collection.first->addRenderTasks(m_textured_glyph_identifiers, textured_shader,
            m_billboard_dummy_mesh, m_textured_gylph_draw_commands, m_textured_glyph_data);
        m_rendertask_collection.second.insert(m_rendertask_collection.second.end(),
            m_textured_glyph_identifiers.begin(), m_textured_glyph_identifiers.end());
    }

    if (!m_scalar_probe_glyph_data.empty()) {
        m_rendertask_collection.first->addRenderTasks(m_scalar_probe_glyph_identifiers, scalar_shader,
            m_billboard_dummy_mesh, m_scalar_probe_gylph_draw_commands, m_scalar_probe_glyph_data);
        m_rendertask_collection.second.insert(m_rendertask_collection.second.end(),
            m_scalar_probe_glyph_identifiers.begin(), m_scalar_probe_glyph_identifiers.end());
    }

    if (!m_scalar_distribution_probe_glyph_data.empty()) {
        m_rendertask_collection.first->addRenderTasks(m_scalar_distribution_probe_glyph_identifiers,
            scalar_distribution_shader, m_billboard_dummy_mesh, m_scalar_distribution_probe_gylph_draw_commands,
            m_scalar_distribution_probe_glyph_data);
        m_rendertask_collection.second.insert(m_rendertask_collection.second.end(),
            m_scalar_distribution_probe_glyph_identifiers.begin(), m_scalar_distribution_probe_glyph_identifiers.end());
    }

    if (!m_vector_probe_glyph_data.empty()) {
        m_rendertask_collection.first->addRenderTasks(m_vector_probe_glyph_identifiers, vector_shader,
            m_billboard_dummy_mesh, m_vector_probe_gylph_draw_commands, m_vector_probe_glyph_data);
        m_rendertask_collection.second.insert(m_rendertask_collection.second.end(),
            m_vector_probe_glyph_identifiers.begin(), m_vector_probe_glyph_identifiers.end());
    }

    if (!m_clusterID_gylph_draw_commands.empty()) {
        m_rendertask_collection.first->addRenderTasks(m_clusterID_glyph_identifiers, clusterID_shader,
            m_billboard_dummy_mesh, m_clusterID_gylph_draw_commands, m_clusterID_glyph_data);
        m_rendertask_collection.second.insert(m_rendertask_collection.second.end(),
            m_clusterID_glyph_identifiers.begin(), m_clusterID_glyph_identifiers.end());
    }
}

void megamol::probe_gl::ProbeBillboardGlyphRenderTasks::updateAllRenderTasks() {
    for (int i = 0; i < m_type_index_map.size(); ++i) {
        auto probe_type = m_type_index_map[i].first;
        auto probe_idx = m_type_index_map[i].second;

        if (probe_type == std::type_index(typeid(GlyphScalarProbeData))) {
            std::array<GlyphScalarProbeData, 1> per_probe_data = {m_scalar_probe_glyph_data[probe_idx]};
            std::string identifier = std::string(FullName()) + "_sg_" + std::to_string(probe_idx);
            m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
        } else if (probe_type == std::type_index(typeid(GlyphVectorProbeData))) {
            std::array<GlyphVectorProbeData, 1> per_probe_data = {m_vector_probe_glyph_data[probe_idx]};
            std::string identifier = std::string(FullName()) + "_vg_" + std::to_string(probe_idx);
            m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
        } else if (probe_type == std::type_index(typeid(GlyphClusterIDData))) {
            std::array<GlyphClusterIDData, 1> per_probe_data = {m_clusterID_glyph_data[probe_idx]};
            std::string identifier = std::string(FullName()) + "_cg_" + std::to_string(probe_idx);
            m_rendertask_collection.first->updatePerDrawData(identifier, per_probe_data);
        }
    }
}

void megamol::probe_gl::ProbeBillboardGlyphRenderTasks::clearAllRenderTasks() {
    for (auto& identifier : m_rendertask_collection.second) {
        m_rendertask_collection.first->deleteRenderTask(identifier);
    }
    m_rendertask_collection.second.clear();
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

    glyph_data.sample_cnt = std::min(static_cast<size_t>(32), probe.getSamplingResult()->samples.size());

    for (int i = 0; i < glyph_data.sample_cnt; ++i) {
        glyph_data.samples[i] = probe.getSamplingResult()->samples[i];
    }

    glyph_data.probe_id = probe_id;

    return glyph_data;
}

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::GlyphScalarDistributionProbeData
megamol::probe_gl::ProbeBillboardGlyphRenderTasks::createScalarDistributionProbeGlyphData(
    probe::FloatDistributionProbe const& probe, int probe_id, float scale) {

    GlyphScalarDistributionProbeData glyph_data;
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
        glyph_data.samples[i][0] = probe.getSamplingResult()->samples[i].mean;
        glyph_data.samples[i][1] = probe.getSamplingResult()->samples[i].lower_bound;
        glyph_data.samples[i][2] = probe.getSamplingResult()->samples[i].upper_bound;
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

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::GlyphClusterIDData
megamol::probe_gl::ProbeBillboardGlyphRenderTasks::createClusterIDGlyphData(
    probe::BaseProbe const& probe, int probe_id, float scale) {

    GlyphClusterIDData glyph_data;
    glyph_data.position = glm::vec4(probe.m_position[0] + probe.m_direction[0] * (probe.m_begin * 1.25f),
        probe.m_position[1] + probe.m_direction[1] * (probe.m_begin * 1.25f),
        probe.m_position[2] + probe.m_direction[2] * (probe.m_begin * 1.25f), 1.0f);

    glyph_data.probe_direction = glm::vec4(probe.m_direction[0], probe.m_direction[1], probe.m_direction[2], 1.0f);

    glyph_data.scale = scale;

    glyph_data.probe_id = probe_id;

    glyph_data.cluster_id = probe.m_cluster_id;

    return glyph_data;
}
