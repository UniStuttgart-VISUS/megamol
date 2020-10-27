#include "ProbeDetailViewRenderTasks.h"

#include "mmcore/EventCall.h"

#include "ProbeCalls.h"
#include "ProbeEvents.h"
#include "ProbeGlCalls.h"
#include "mesh/MeshCalls.h"
#include "mmcore/view/CallGetTransferFunction.h"

megamol::probe_gl::ProbeDetailViewRenderTasks::ProbeDetailViewRenderTasks()
        : AbstractGPURenderTaskDataSource()
        , m_version(0)
        , m_transfer_function_Slot("GetTransferFunction", "Slot for accessing a transfer function")
        , m_probes_slot("GetProbes", "Slot for accessing a probe collection")
        , m_event_slot("GetProbeManipulation", "")
        , m_ui_mesh(nullptr)
        , m_probes_mesh(nullptr)
        , m_tf_min(0.0f)
        , m_tf_max(1.0f) {
    this->m_transfer_function_Slot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->m_transfer_function_Slot);

    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_event_slot.SetCompatibleCall<core::CallEventDescription>();
    this->MakeSlotAvailable(&this->m_event_slot);
}

megamol::probe_gl::ProbeDetailViewRenderTasks::~ProbeDetailViewRenderTasks() {}

bool megamol::probe_gl::ProbeDetailViewRenderTasks::create() {

    AbstractGPURenderTaskDataSource::create();

    // TODO ui mesh
    {
        // clang-format off
        std::vector<std::vector<float>> vertices = {{
                -1.0, -0.05, 0.0,
                1.0, 0.05, 0.0,
                -1.0, 0.05, 0.0,
                1.0, 0.05, 0.0,
                -1.0, -0.05, 0.0,
                1.0, -0.05, 0.0,

                0.0, -0.05, -1.0,
                0.0, 0.05, 1.0,
                0.0, 0.05, -1.0,
                0.0, 0.05, 1.0,
                0.0, -0.05, -1.0,
                0.0, -0.05, 1.0
            }};
        std::vector<size_t> byte_sizes = {};
        std::vector<uint32_t> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        std::vector<glowl::VertexLayout> vertex_layout = {
            glowl::VertexLayout(12,{glowl::VertexLayout::Attribute(3,GL_FLOAT,GL_FALSE,0)})
        };
        try {
            m_probes_mesh = std::make_shared<glowl::Mesh>(vertices, indices, vertex_layout,
                GL_UNSIGNED_INT, GL_STATIC_DRAW, GL_TRIANGLES);
        } catch (const std::exception&) {
        }
    }
    

    // create an empty dummy mesh, probe mesh
    std::vector<void*> data_ptrs = {};
    std::vector<size_t> byte_sizes = {};
    std::vector<uint32_t> indices = {0, 1, 2, 3, 4, 5};
    std::vector<glowl::VertexLayout> vertex_layout = {};

    m_probes_mesh = std::make_shared<glowl::Mesh>(
        data_ptrs, byte_sizes, indices.data(), 6 * 4, vertex_layout, GL_UNSIGNED_INT, GL_STATIC_DRAW, GL_TRIANGLES);

    return true; 
}

void megamol::probe_gl::ProbeDetailViewRenderTasks::release() {}

bool megamol::probe_gl::ProbeDetailViewRenderTasks::getDataCallback(core::Call& caller) {

    mesh::CallGPURenderTaskData* lhs_rtc = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL) return false;

    syncRenderTaskCollection(lhs_rtc);

    // if there is a render task connection to the right, pass on the render task collection
    mesh::CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<mesh::CallGPURenderTaskData>();
    if (rhs_rtc != NULL) {
        rhs_rtc->setData(m_rendertask_collection.first, 0);
        if (!(*rhs_rtc)(0)) return false;
    }

    // check/get material 
    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
    if (mtlc == NULL) return false;
    if (!(*mtlc)(0)) return false;

    // check/get mesh data 
    mesh::CallGPUMeshData* mc = this->m_mesh_slot.CallAs<mesh::CallGPUMeshData>();
    if (mc != NULL){
        if (!(*mc)(0)) return false;
    }

    // check/get probe data
    probe::CallProbes* pc = this->m_probes_slot.CallAs<probe::CallProbes>();
    if (pc == NULL) return false;
    if (!(*pc)(0)) return false;

    // check/get transfer function
    auto* tfc = this->m_transfer_function_Slot.CallAs<core::view::CallGetTransferFunction>();
    if (tfc != NULL) {
        if (!(*tfc)(0)) return false;
    }

    bool something_has_changed = pc->hasUpdate() || mtlc->hasUpdate() || mc->hasUpdate() || ((tfc != NULL) ? tfc->IsDirty() : false);

    if (something_has_changed) {
        ++m_version;

        auto probes = pc->getData();
        auto probe_cnt = probes->getProbeCount();

        m_vector_probe_draw_commands.clear();
        m_vector_probe_draw_commands.reserve(probe_cnt);

        m_vector_probe_data.clear();
        m_vector_probe_data.reserve(probe_cnt);

        std::vector<std::vector<float>> vertex_data;
        vertex_data.push_back(std::vector<float>());
        vertex_data.back().reserve(3 * 2 * probe_cnt * 32); //TODO generic sample count per probe...

        std::vector<uint32_t> index_data;
        index_data.reserve(4 * (probe_cnt-1));

        for (int probe_idx = 0; probe_idx < probe_cnt; ++probe_idx) {

            auto generic_probe = probes->getGenericProbe(probe_idx);

            auto visitor = [&vertex_data, &index_data, probe_idx, this](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, probe::FloatProbe>) {
                    // TODO
                } else if constexpr (std::is_same_v<T, probe::IntProbe>) {
                    // TODO
                } else if constexpr (std::is_same_v<T, probe::Vec4Probe>) {

                    auto probe_length = arg.m_end - arg.m_begin;
                    glm::vec3 center_point = glm::vec3(0.0f,-arg.m_begin,0.0f);
                
                    auto base_vertex_idx = vertex_data.back().size() / 3u;
                
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
                
                    m_vector_probe_data.push_back(probe_data);
                    
                
                    // generate vertices
                    for (auto& sample : sample_data->samples )
                    {
                        vertex_data.back().push_back(center_point.x - 0.5f * std::get<0>(sample));
                        vertex_data.back().push_back(center_point.x - 0.5f * std::get<1>(sample));
                        vertex_data.back().push_back(center_point.x - 0.5f * std::get<2>(sample));
                
                        vertex_data.back().push_back(center_point.x + 0.5f * std::get<0>(sample));
                        vertex_data.back().push_back(center_point.x + 0.5f * std::get<1>(sample));
                        vertex_data.back().push_back(center_point.x + 0.5f * std::get<2>(sample));
                
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
                       index_data.push_back((i*2) + 2);
                       index_data.push_back((i*2) + 3);
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

                } else {
                    // unknown probe type, throw error? do nothing?
                }
            };

            std::visit(visitor, generic_probe);
        }

        std::vector<glowl::VertexLayout> vertex_layout = {
            glowl::VertexLayout(12,{glowl::VertexLayout::Attribute(3,GL_FLOAT,GL_FALSE,0)})
        };
        try {
            m_probes_mesh = std::make_shared<glowl::Mesh>(vertex_data, index_data, vertex_layout,
                GL_UNSIGNED_INT, GL_STATIC_DRAW, GL_PATCHES);
        } catch (const std::exception&) {
        }

        auto gpu_mtl_storage = mtlc->getData();

        if (!m_vector_probe_draw_commands.empty()) {
            auto const& probe_shader = gpu_mtl_storage->getMaterial("ProbeDetailView_Ribbon").shader_program;
            m_rendertask_collection.first->addRenderTasks(m_vector_probe_identifiers,probe_shader, m_probes_mesh, m_vector_probe_draw_commands, m_vector_probe_data);
            m_rendertask_collection.second.insert(
                            m_rendertask_collection.second.end(), m_vector_probe_identifiers.begin(), m_vector_probe_identifiers.end());
        }

    }

    //TODO set data/metadata

    // check for pending probe manipulations
    // check for pending events
    auto call_event_storage = this->m_event_slot.CallAs<core::CallEvent>();
    if (call_event_storage != NULL) {
        if ((!(*call_event_storage)(0))) return false;

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

            }
        }

        // process probe selection toggle events
        {
            auto pending_select_events = event_collection->get<ProbeSelectToggle>();
            for (auto& evt : pending_select_events) {
                
            }
        }

    }

    return true;
}

bool megamol::probe_gl::ProbeDetailViewRenderTasks::getMetaDataCallback(core::Call& caller) {
    if (!AbstractGPURenderTaskDataSource::getMetaDataCallback(caller))
        return false;

    mesh::CallGPURenderTaskData* lhs_rt_call = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
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


