#include "ProbeDetailViewRenderer.h"

#include "mmstd/event/EventCall.h"

#include "ProbeEvents.h"
#include "ProbeGlCalls.h"
#include "mesh/MeshCalls.h"
#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"
#include "probe/ProbeCalls.h"

megamol::probe_gl::ProbeDetailViewRenderer::ProbeDetailViewRenderer()
        : m_transfer_function_Slot("GetTransferFunction", "Slot for accessing a transfer function")
        , m_probes_slot("GetProbes", "Slot for accessing a probe collection")
        , m_event_slot("GetProbeManipulation", "")
        , m_ui_mesh(nullptr)
        , m_probes_mesh(nullptr)
        , m_tf_min(0.0f)
        , m_tf_max(1.0f) {
    this->m_transfer_function_Slot.SetCompatibleCall<mmstd_gl::CallGetTransferFunctionGLDescription>();
    this->MakeSlotAvailable(&this->m_transfer_function_Slot);

    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_event_slot.SetCompatibleCall<core::CallEventDescription>();
    this->MakeSlotAvailable(&this->m_event_slot);
}

megamol::probe_gl::ProbeDetailViewRenderer::~ProbeDetailViewRenderer() {}

void megamol::probe_gl::ProbeDetailViewRenderer::createMaterialCollection() {
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
        // TODO intialize with value indicating that no transfer function is connected
        this->m_transfer_function->makeResident();
    } catch (glowl::TextureException const& exc) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error on transfer texture view pre-creation: %s. [%s, %s, line %d]\n", exc.what(), __FILE__, __FUNCTION__,
            __LINE__);
    }

    material_collection_ = std::make_shared<mesh_gl::GPUMaterialCollection>();
    try {
        std::vector<std::filesystem::path> shaderfiles = {"probes/dfr_probeDetailView.vert.glsl",
            "probes/dfr_probeDetailView.frag.glsl", "probes/dfr_probeDetailView.tesc.glsl",
            "probes/dfr_probeDetailView.tese.glsl"};
        material_collection_->addMaterial(this->instance(), "ProbeDetailView", shaderfiles);
    } catch (const std::exception& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "%s [%s, %s, line %d]\n", ex.what(), __FILE__, __FUNCTION__, __LINE__);
    }

    // TODO ui mesh
    {
        // clang-format off
        std::vector<std::vector<float>> vertices = {{
                -5.0, -0.05, 0.0,
                5.0, 0.05, 0.0,
                -5.0, 0.05, 0.0,
                5.0, 0.05, 0.0,
                -5.0, -0.05, 0.0,
                5.0, -0.05, 0.0,

                0.0, -0.05, -5.0,
                0.0, 0.05, 5.0,
                0.0, 0.05, -5.0,
                0.0, 0.05, 5.0,
                0.0, -0.05, -5.0,
                0.0, -0.05, 5.0
            }};
        std::vector<size_t> byte_sizes = {};
        std::vector<uint32_t> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        std::vector<glowl::VertexLayout> vertex_layout = {
            glowl::VertexLayout(12,{glowl::VertexLayout::Attribute(3,GL_FLOAT,GL_FALSE,0)})
        };
        try {
            m_ui_mesh = std::make_shared<glowl::Mesh>(vertices, vertex_layout, indices,
                GL_UNSIGNED_INT, GL_TRIANGLES, GL_STATIC_DRAW);
        } catch (const std::exception& exc) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error on UI mesh creation: %s. [%s, %s, line %d]\n", exc.what(), __FILE__, __FUNCTION__,
            __LINE__);
        }
    }

    // create an empty dummy mesh, probe mesh
    std::vector<void const*> data_ptrs = {};
    std::vector<size_t> byte_sizes = {};
    std::vector<uint32_t> indices = {0, 1, 2, 3, 4, 5};
    std::vector<glowl::VertexLayout> vertex_layout = {};

    //TODO try catch
    m_probes_mesh = std::make_shared<glowl::Mesh>(
        data_ptrs, byte_sizes, vertex_layout, indices.data(), 6 * 4, GL_UNSIGNED_INT, GL_TRIANGLES, GL_STATIC_DRAW);
}

void megamol::probe_gl::ProbeDetailViewRenderer::updateRenderTaskCollection(
    mmstd_gl::CallRender3DGL& call, bool force_update) {

    // check/get probe data
    probe::CallProbes* pc = this->m_probes_slot.CallAs<probe::CallProbes>();
    if (pc != nullptr){
    
    if (!(*pc)(0)){
        // TODO throw error
        return;
    }
    // check/get transfer function
    auto* tfc = this->m_transfer_function_Slot.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
    if (tfc != nullptr) {
        if (!(*tfc)(0)){
        // TODO throw error
        return;
    }
    }

    bool something_has_changed = pc->hasUpdate() || ((tfc != NULL) ? tfc->IsDirty() : false);

    if (something_has_changed) {
        render_task_collection_->clear();

        auto const& ui_shader = material_collection_->getMaterial("ProbeDetailViewUI").shader_program;
        std::vector<glowl::DrawElementsCommand> draw_commands = {glowl::DrawElementsCommand()};
        draw_commands[0].cnt = 12;
        draw_commands[0].instance_cnt = 1;
        draw_commands[0].first_idx = 0;
        draw_commands[0].base_vertex = 0;
        draw_commands[0].base_instance = 0;
        struct UIPerDrawData {};
        std::vector<UIPerDrawData> per_draw_data = {UIPerDrawData()};
        std::vector<std::string> identifiers = {std::string(FullName()) + "UI"};
        render_task_collection_->addRenderTasks(identifiers, ui_shader, m_ui_mesh, draw_commands, per_draw_data );

        auto probes = pc->getData();
        auto probe_cnt = probes->getProbeCount();

        m_vector_probe_identifiers.clear();
        m_vector_probe_identifiers.reserve(probe_cnt);

        m_vector_probe_selected.clear();
        m_vector_probe_selected.reserve(probe_cnt);

        m_vector_probe_draw_commands.clear();
        m_vector_probe_draw_commands.reserve(probe_cnt);

        m_vector_probe_data.clear();
        m_vector_probe_data.reserve(probe_cnt);

        std::vector<std::vector<float>> vertex_data;
        vertex_data.push_back(std::vector<float>());
        vertex_data.back().reserve(4 * 2 * probe_cnt * 32); //TODO generic sample count per probe...

        std::vector<uint32_t> index_data;
        index_data.reserve(4 * (probe_cnt-1));

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
                tex_layout.int_parameters = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST},
                    {GL_TEXTURE_MAG_FILTER, GL_LINEAR}, {GL_TEXTURE_WRAP_S, GL_CLAMP}};
                try {
                    this->m_transfer_function = std::make_shared<glowl::Texture2D>(
                        "ProbeTransferFunction", tex_layout, (GLvoid*) tfc->GetTextureData());
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
                            "Error on making transfer texture view resident: %i. [%s, %s, line %d]\n", err,
                            __FILE__, __FUNCTION__, __LINE__);
                    }
                }

                m_tf_min = std::get<0>(tfc->Range());
                m_tf_max = std::get<1>(tfc->Range());
            }
        }

        GLuint64 texture_handle = this->m_transfer_function->getTextureHandle();

        //for (int probe_idx = 0; probe_idx < std::min(probe_cnt,1u); ++probe_idx) {
        for (int probe_idx = 0; probe_idx < probe_cnt; ++probe_idx) {

            auto generic_probe = probes->getGenericProbe(probe_idx);

            auto visitor = [&vertex_data, &index_data, probe_idx, texture_handle, this](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, probe::FloatProbe>) {
                    // TODO
                } else if constexpr (std::is_same_v<T, probe::IntProbe>) {
                    // TODO
                } else if constexpr (std::is_same_v<T, probe::Vec4Probe>) {

                    auto probe_length = arg.m_end - arg.m_begin;
                    glm::vec3 center_point = glm::vec3(0.0f,-arg.m_begin,0.0f);
                
                    auto base_vertex_idx = vertex_data.back().size() / 4u;
                
                    auto sample_data = arg.getSamplingResult();
                    size_t sample_cnt = sample_data->samples.size();
                
                
                    VectorProbeData probe_data;
                    probe_data.position = glm::vec4(arg.m_position[0] + arg.m_direction[0] * (arg.m_begin * 1.25f),
                        arg.m_position[1] + arg.m_direction[1] * (arg.m_begin * 1.25f),
                        arg.m_position[2] + arg.m_direction[2] * (arg.m_begin * 1.25f), 1.0f);
                    probe_data.probe_direction = glm::vec4(arg.m_direction[0], arg.m_direction[1], arg.m_direction[2], 1.0f);
                    probe_data.scale = 1.0f; //TODO scale?

                    assert(sample_cnt <= 32);

                    probe_data.sample_cnt = std::min(static_cast<size_t>(32), sample_cnt);
                    for (int i = 0; i < probe_data.sample_cnt; ++i) {
                        probe_data.samples[i] = sample_data->samples[i];
                    }
                    probe_data.probe_id = probe_idx;

                    probe_data.tf_texture_handle = texture_handle;
                    probe_data.tf_max = m_tf_max;
                    probe_data.tf_min = m_tf_min;
                
                    m_vector_probe_data.push_back(probe_data);
                    
                    float ribbon_width = 2.0f;

                    // generate vertices
                    for (auto& sample : sample_data->samples )
                    {
                        std::array<float,4> sample_normalized = sample;
                        float l = std::sqrt(sample_normalized[0]*sample_normalized[0] + sample_normalized[1]*sample_normalized[1] + sample_normalized[2]*sample_normalized[2]);
                        sample_normalized[0] /= l;
                        sample_normalized[1] /= l;
                        sample_normalized[2] /= l;

                        vertex_data.back().push_back(center_point.x - 0.5f * ribbon_width * std::get<0>(sample_normalized));
                        vertex_data.back().push_back(center_point.y - 0.5f * ribbon_width * std::get<1>(sample_normalized));
                        vertex_data.back().push_back(center_point.z - 0.5f * ribbon_width * std::get<2>(sample_normalized));
                        vertex_data.back().push_back(sample[3]);

                        vertex_data.back().push_back(center_point.x + 0.5f * ribbon_width * std::get<0>(sample_normalized));
                        vertex_data.back().push_back(center_point.y + 0.5f * ribbon_width * std::get<1>(sample_normalized));
                        vertex_data.back().push_back(center_point.z + 0.5f * ribbon_width * std::get<2>(sample_normalized));
                        vertex_data.back().push_back(sample[3]);

                        center_point += glm::vec3(
                            0.0f,
                            probe_length / static_cast<float>(sample_cnt),
                            0.0f);
                    }

                    auto first_index_idx = index_data.size();
                    // generate indices
                    auto patch_cnt = std::max(0,(static_cast<int>(sample_cnt) - 1));
                    for (int i = 0; i < patch_cnt; ++i)
                    {
                       index_data.push_back((i*2) + 0);
                       index_data.push_back((i*2) + 1);
                       index_data.push_back((i*2) + 3);
                       index_data.push_back((i*2) + 2);
                    }
                
                    //TODO set correct values...
                    glowl::DrawElementsCommand draw_command;
                    draw_command.base_instance = 0;
                    draw_command.base_vertex = base_vertex_idx;
                    draw_command.cnt = patch_cnt * 4;
                    draw_command.first_idx = first_index_idx;
                    draw_command.instance_cnt = 1;
                
                    m_vector_probe_draw_commands.push_back(draw_command);
                
                    m_vector_probe_identifiers.emplace_back(std::string(FullName())+"_probe_"+std::to_string(probe_idx));
                    m_vector_probe_selected.push_back(false);

                } else {
                    // unknown probe type, throw error? do nothing?
                }
            };

            std::visit(visitor, generic_probe);
        }

        std::vector<glowl::VertexLayout> vertex_layout = {
            glowl::VertexLayout(16,{glowl::VertexLayout::Attribute(4,GL_FLOAT,GL_FALSE,0)})
        };
        try {
            m_probes_mesh = std::make_shared<glowl::Mesh>(vertex_data, vertex_layout, index_data,
                GL_UNSIGNED_INT, GL_PATCHES, GL_STATIC_DRAW);
        } catch (glowl::MeshException const& exc) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error during mesh creation of\"%s\": %s. [%s, %s, line %d]\n", "ProbeDetailVieW",
            exc.what(), __FILE__, __FUNCTION__, __LINE__);
        }
        
        if (!m_vector_probe_draw_commands.empty()) {
            auto const& probe_shader = material_collection_->getMaterial("ProbeDetailView").shader_program;
            render_task_collection_->addRenderTasks(m_vector_probe_identifiers,probe_shader, m_probes_mesh, m_vector_probe_draw_commands, m_vector_probe_data);
        }
    }

    //TODO set data/metadata
    // compute mesh call specific update
    std::array<float, 6> bbox;

    bbox[0] = -3.0f;
    bbox[1] = 64.0f;
    bbox[2] = -3.0f;
    bbox[3] = 3.0f;
    bbox[4] = -0.5f;
    bbox[5] = 3.0f;

    call.AccessBoundingBoxes().SetBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
    call.AccessBoundingBoxes().SetClipBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);    

    // check for pending probe manipulations
    // check for pending events
    auto call_event_storage = this->m_event_slot.CallAs<core::CallEvent>();
    if (call_event_storage != NULL) {
        if ((!(*call_event_storage)(0))){
            // TODO throw
            return;
        }

        auto event_collection = call_event_storage->getData();

        // process pobe clear selection events
        {
            auto pending_clearselection_events = event_collection->get<ProbeClearSelection>();
            for (auto& evt : pending_clearselection_events) {
               
            }
        }

        // process probe highlight events
        {
            auto pending_highlight_events = event_collection->get<ProbeHighlight>();
            for (auto& evt : pending_highlight_events) {

            }
        }

        // process probe dehighlight events
        {
            auto pending_dehighlight_events = event_collection->get<ProbeDehighlight>();
            for (auto& evt : pending_dehighlight_events) {
                
            }
        }

        // process probe selection events
        {
            auto pending_select_events = event_collection->get<ProbeSelect>();
            for (auto& evt : pending_select_events) {
                
            }
        }

        // process probe deselection events
        {
            auto pending_deselect_events = event_collection->get<ProbeDeselect>();
            for (auto& evt : pending_deselect_events) {
                
            }
        }

        // process probe exclusive selection events
        {
            auto pending_events = event_collection->get<ProbeSelectExclusive>();
            if (!pending_events.empty()) {
                auto probe_idx = pending_events.back().obj_id;
                
                render_task_collection_->clear();

                for (int i=0; i< m_vector_probe_selected.size(); ++i) {
                    m_vector_probe_selected[i] = false;
                }
                
                if (!m_vector_probe_draw_commands.empty()) {
                    auto const& probe_shader = material_collection_->getMaterial("ProbeDetailView").shader_program;
                    std::string identifier = std::string(FullName()) + std::to_string(probe_idx);
                    render_task_collection_->addRenderTask(identifier,
                    probe_shader, m_probes_mesh, m_vector_probe_draw_commands[probe_idx], m_vector_probe_data[probe_idx]);

                    m_vector_probe_selected[probe_idx] = true;
                }

                auto const& ui_shader = material_collection_->getMaterial("ProbeDetailViewUI").shader_program;
                std::vector<glowl::DrawElementsCommand> draw_commands = {glowl::DrawElementsCommand()};
                draw_commands[0].cnt = 12;
                draw_commands[0].instance_cnt = 1;
                draw_commands[0].first_idx = 0;
                draw_commands[0].base_vertex = 0;
                draw_commands[0].base_instance = 0;
                struct UIPerDrawData {};
                std::vector<UIPerDrawData> per_draw_data = {UIPerDrawData()};
                std::vector<std::string> identifiers = {std::string(FullName()) + "UI"};
                render_task_collection_->addRenderTasks(identifiers, ui_shader, m_ui_mesh, draw_commands, per_draw_data );
            }
        }

        // process probe selection toggle events
        {
            auto pending_select_events = event_collection->get<ProbeSelectToggle>();
            for (auto& evt : pending_select_events) {
                auto probe_idx = evt.obj_id;

                if (!m_vector_probe_draw_commands.empty()) {
                    std::string identifier = std::string(FullName()) + std::to_string(probe_idx);
                    if (!m_vector_probe_selected[probe_idx]) {
                        auto const& probe_shader = material_collection_->getMaterial("ProbeDetailView").shader_program;

                        render_task_collection_->addRenderTask(identifier,
                        probe_shader, m_probes_mesh, m_vector_probe_draw_commands[probe_idx], m_vector_probe_data[probe_idx]);

                        m_vector_probe_selected[probe_idx] = true;
                    }
                    else {
                        render_task_collection_->deleteRenderTask(identifier);
                        m_vector_probe_selected[probe_idx] = false;
                    }
                }
            }
        }

    }
    }
}

bool megamol::probe_gl::ProbeDetailViewRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
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


