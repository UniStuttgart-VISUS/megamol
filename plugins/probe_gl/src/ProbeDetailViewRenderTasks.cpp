#include "ProbeDetailViewRenderTasks.h"

#include "mmcore/EventCall.h"

#include "ProbeCalls.h"
#include "ProbeEvents.h"
#include "ProbeGlCalls.h"
#include "mesh/MeshCalls.h"
#include "mmcore/view/CallGetTransferFunction.h"

megamol::probe_gl::ProbeDetailViewRenderTasks::ProbeDetailViewRenderTasks()
    : m_version(0)
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

    //TODO ui mesh
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

    return false; 
}

void megamol::probe_gl::ProbeDetailViewRenderTasks::release() {}

bool megamol::probe_gl::ProbeDetailViewRenderTasks::getDataCallback(core::Call& caller) {

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

    // check/get material 
    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
    if (mtlc == NULL) return false;
    if (!(*mtlc)(0)) return false;

    // check/get mesh data 
    mesh::CallGPUMeshData* mc = this->m_mesh_slot.CallAs<mesh::CallGPUMeshData>();
    if (mc == NULL) return false;
    if (!(*mc)(0)) return false;

    // check/get probe data
    probe::CallProbes* pc = this->m_probes_slot.CallAs<probe::CallProbes>();
    if (pc == NULL) return false;
    if (!(*pc)(0)) return false;

    // check/get transfer function
    auto* tfc = this->m_transfer_function_Slot.CallAs<core::view::CallGetTransferFunction>();
    if (tfc != NULL) {
        ((*tfc)(0));
    }

    bool something_has_changed = pc->hasUpdate() || mtlc->hasUpdate() || mc->hasUpdate() || ((tfc != NULL) ? tfc->IsDirty() : false);

    if (something_has_changed) {
        ++m_version;


        //TODO everything

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

    return false;
}

bool megamol::probe_gl::ProbeDetailViewRenderTasks::getMetaDataCallback(core::Call& caller) { return false; }

megamol::probe_gl::ProbeDetailViewRenderTasks::VectorProbeData
megamol::probe_gl::ProbeDetailViewRenderTasks::createVectorProbeData(
    probe::Vec4Probe const& probe, int probe_id, float scale) {
    return VectorProbeData();
}
