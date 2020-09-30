#include "ProbeDetailViewRenderTasks.h"

#include "ProbeCalls.h"
#include "ProbeGlCalls.h"
#include "mesh/MeshCalls.h"
#include "mmcore/view/CallGetTransferFunction.h"

megamol::probe_gl::ProbeDetailViewRenderTasks::ProbeDetailViewRenderTasks()
    : m_version(0)
    , m_transfer_function_Slot("GetTransferFunction", "Slot for accessing a transfer function")
    , m_probes_slot("GetProbes", "Slot for accessing a probe collection")
    , m_probe_manipulation_slot("GetProbeManipulation", "")
    , m_ui_mesh(nullptr)
    , m_probes_mesh(nullptr)
    , m_tf_min(0.0f)
    , m_tf_max(1.0f) {
    this->m_transfer_function_Slot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->m_transfer_function_Slot);

    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_probe_manipulation_slot.SetCompatibleCall<probe_gl::CallProbeInteractionDescription>();
    this->MakeSlotAvailable(&this->m_probe_manipulation_slot);
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

    // check/get probe data
    probe::CallProbes* pc = this->m_probes_slot.CallAs<probe::CallProbes>();
    if (pc == NULL) return false;
    if (!(*pc)(0)) return false;

    // check/get transfer function
    auto* tfc = this->m_transfer_function_Slot.CallAs<core::view::CallGetTransferFunction>();
    if (tfc != NULL) {
        ((*tfc)(0));
    }

    bool something_has_changed = pc->hasUpdate() || mtlc->hasUpdate() || ((tfc != NULL) ? tfc->IsDirty() : false);

    if (something_has_changed) {
        ++m_version;


        //TODO everything

    }

    //TODO set data/metadata

    // check for pending probe manipulations
    CallProbeInteraction* pic = this->m_probe_manipulation_slot.CallAs<CallProbeInteraction>();
    if (pic != NULL) {
        if (!(*pic)(0)) return false;

        if (pic->hasUpdate()) {
            auto interaction_collection = pic->getData();

            auto& pending_manips = interaction_collection->accessPendingManipulations();

            if (pc->hasUpdate()) {
                if (!(*pc)(0)) return false;
            }
            auto probes = pc->getData();

            for (auto itr = pending_manips.begin(); itr != pending_manips.end(); ++itr) {
                if (itr->type == HIGHLIGHT) {
                    auto manipulation = *itr;


                } else if (itr->type == DEHIGHLIGHT) {
                    auto manipulation = *itr;


                } else if (itr->type == SELECT) {
                    auto manipulation = *itr;

                    //TODO add render tasks
                } else if (itr->type == DESELECT) {
                    auto manipulation = *itr;

                    //TODO remove render tasks
                } else if (itr->type == CLEAR_SELECTION) {
                    auto manipulation = *itr;

                    //TODO clear render tasks
                } else {
                }
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
