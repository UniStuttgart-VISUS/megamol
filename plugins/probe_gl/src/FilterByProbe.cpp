#include "FilterByProbe.h"

#include "ProbeEvents.h"
#include "ProbeGlCalls.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/EventCall.h"
#include "mmcore_gl/FlagCallsGL.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"
#include "probe/CallKDTree.h"
#include "probe/ProbeCalls.h"

#include "probe/ProbeCollection.h"

namespace megamol {
namespace probe_gl {

FilterByProbe::FilterByProbe()
        : m_version(0)
        , m_probe_selection()
        , m_probes_slot("getProbes", "")
        , m_kd_tree_slot("getKDTree", "")
        , m_event_slot("getEvents", "")
        , m_readFlagsSlot("getReadFlags", "")
        , m_writeFlagsSlot("getWriteFlags", "") {

    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_kd_tree_slot.SetCompatibleCall<probe::CallKDTreeDescription>();
    this->MakeSlotAvailable(&this->m_kd_tree_slot);

    this->m_event_slot.SetCompatibleCall<megamol::core::CallEventDescription>();
    this->MakeSlotAvailable(&this->m_event_slot);

    this->m_readFlagsSlot.SetCompatibleCall<core_gl::FlagCallRead_GLDescription>();
    this->MakeSlotAvailable(&this->m_readFlagsSlot);

    this->m_writeFlagsSlot.SetCompatibleCall<core_gl::FlagCallWrite_GLDescription>();
    this->MakeSlotAvailable(&this->m_writeFlagsSlot);
}

FilterByProbe::~FilterByProbe() {
    this->Release();
}

bool FilterByProbe::create() {
    try {
        // create shader program
        m_setFlags_prgm = std::make_unique<GLSLComputeShader>();
        m_filterAll_prgm = std::make_unique<GLSLComputeShader>();
        m_filterNone_prgm = std::make_unique<GLSLComputeShader>();

        vislib_gl::graphics::gl::ShaderSource setFlags_src;
        vislib_gl::graphics::gl::ShaderSource filterAll_src;
        vislib_gl::graphics::gl::ShaderSource filterNone_src;

        auto ssf =
            std::make_shared<core_gl::utility::ShaderSourceFactory>(instance()->Configuration().ShaderDirectories());

        if (!ssf->MakeShaderSource("FilterByProbe::setFlags", setFlags_src))
            return false;
        if (!m_setFlags_prgm->Compile(setFlags_src.Code(), setFlags_src.Count()))
            return false;
        if (!m_setFlags_prgm->Link())
            return false;

        if (!ssf->MakeShaderSource("FilterByProbe::filterAll", filterAll_src))
            return false;
        if (!m_filterAll_prgm->Compile(filterAll_src.Code(), filterAll_src.Count()))
            return false;
        if (!m_filterAll_prgm->Link())
            return false;

        if (!ssf->MakeShaderSource("FilterByProbe::filterNone", filterNone_src))
            return false;
        if (!m_filterNone_prgm->Compile(filterNone_src.Code(), filterNone_src.Count()))
            return false;
        if (!m_filterNone_prgm->Link())
            return false;

    } catch (vislib_gl::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to compile shader (@%s): %s\n",
            vislib_gl::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader: Unknown exception\n");
        return false;
    }

    return true;
}

void FilterByProbe::release() {

    m_setFlags_prgm.reset();
    m_filterAll_prgm.reset();
}

bool FilterByProbe::GetExtents(core_gl::view::CallRender3DGL& call) {
    return true;
}

bool FilterByProbe::Render(core_gl::view::CallRender3DGL& call) {
    probe::CallProbes* pc = this->m_probes_slot.CallAs<probe::CallProbes>();
    if (pc == NULL)
        return false;
    if (!(*pc)(0))
        return false;

    if (pc->hasUpdate()) {
        auto probes = pc->getData();

        m_probe_selection.resize(probes->getProbeCount());
    }

    // query kd tree data
    auto ct = this->m_kd_tree_slot.CallAs<probe::CallKDTree>();
    if (ct == nullptr)
        return false;
    if (!(*ct)(0))
        return false;

    // check for pending events
    auto call_event_storage = this->m_event_slot.CallAs<core::CallEvent>();
    if (call_event_storage != NULL) {
        if ((!(*call_event_storage)(0)))
            return false;

        auto event_collection = call_event_storage->getData();
        auto probes = pc->getData();

        // process pobe clear selection events
        {
            auto pending_clearselection_events = event_collection->get<ProbeClearSelection>();
            if (!pending_clearselection_events.empty()) {
                std::fill(m_probe_selection.begin(), m_probe_selection.end(), false);
            }
        }

        // process probe selection events
        {
            auto pending_select_events = event_collection->get<ProbeSelect>();
            for (auto& evt : pending_select_events) {
                m_probe_selection[evt.obj_id] = true;
            }
        }

        // process probe deselection events
        {
            auto pending_deselect_events = event_collection->get<ProbeDeselect>();
            for (auto& evt : pending_deselect_events) {
                m_probe_selection[evt.obj_id] = false;
            }
        }

        // process probe exclusive selection events
        {
            auto pending_selectExclusive_events = event_collection->get<ProbeSelectExclusive>();
            if (!pending_selectExclusive_events.empty()) {
                std::fill(m_probe_selection.begin(), m_probe_selection.end(), false);
                m_probe_selection[pending_selectExclusive_events.back().obj_id] = true;
            }
        }

        // process probe selection toggle events
        {
            auto pending_select_events = event_collection->get<ProbeSelectToggle>();
            for (auto& evt : pending_select_events) {
                m_probe_selection[evt.obj_id] = m_probe_selection[evt.obj_id] == true ? false : true;
            }
        }

        // process clear filter events
        {
            auto pending_clearselection_events = event_collection->get<DataClearFilter>();

            for (auto& evt : pending_clearselection_events) {
                auto readFlags = m_readFlagsSlot.CallAs<core_gl::FlagCallRead_GL>();
                auto writeFlags = m_writeFlagsSlot.CallAs<core_gl::FlagCallWrite_GL>();

                if (readFlags != nullptr && writeFlags != nullptr) {
                    (*readFlags)(core_gl::FlagCallWrite_GL::CallGetData);

                    if (readFlags->hasUpdate()) {
                        this->m_version = readFlags->version();
                    }

                    ++m_version;

                    auto flag_data = readFlags->getData();

                    {
                        m_filterNone_prgm->Enable();

                        auto flag_cnt = static_cast<GLuint>(flag_data->flags->getByteSize() / sizeof(GLuint));

                        glUniform1ui(m_filterNone_prgm->ParameterLocation("flag_cnt"), flag_cnt);

                        flag_data->flags->bind(1);

                        m_filterNone_prgm->Dispatch(static_cast<int>(std::ceil(flag_cnt / 64.0f)), 1, 1);

                        ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                        m_filterNone_prgm->Disable();
                    }

                    writeFlags->setData(readFlags->getData(), m_version);
                    (*writeFlags)(core_gl::FlagCallWrite_GL::CallGetData);
                }
            }
        }

        // process probe selection events
        {
            auto pending_filter_event = event_collection->get<DataFilterByProbeSelection>();

            if (!pending_filter_event.empty()) {

                // TODO get corresponding data points from kd-tree
                auto tree = ct->getData();
                std::vector<uint32_t> indices;

                for (size_t probe_idx = 0; probe_idx < m_probe_selection.size(); ++probe_idx) {

                    if (m_probe_selection[probe_idx] == true) {
                        // TODO get probe
                        auto generic_probe = probes->getGenericProbe(probe_idx);

                        auto visitor = [&tree, &indices](auto&& arg) {
                            using T = std::decay_t<decltype(arg)>;
                            if constexpr (std::is_same_v<T, probe::Vec4Probe> || std::is_same_v<T, probe::FloatProbe>) {
                                auto position = arg.m_position;
                                auto direction = arg.m_direction;
                                auto begin = arg.m_begin;
                                auto end = arg.m_end;
                                auto samples_per_probe = arg.getSamplingResult()->samples.size();

                                auto sample_step = end / static_cast<float>(samples_per_probe);
                                auto radius = sample_step * 2.0; // sample_radius_factor;

                                for (int j = 0; j < samples_per_probe; j++) {

                                    pcl::PointXYZ sample_point;
                                    sample_point.x = position[0] + j * sample_step * direction[0];
                                    sample_point.y = position[1] + j * sample_step * direction[1];
                                    sample_point.z = position[2] + j * sample_step * direction[2];

                                    std::vector<float> k_distances;
                                    std::vector<uint32_t> k_indices;

                                    auto num_neighbors =
                                        tree->radiusSearch(sample_point, arg.m_sample_radius, k_indices, k_distances);
                                    if (num_neighbors == 0) {
                                        num_neighbors = tree->nearestKSearch(sample_point, 1, k_indices, k_distances);
                                    }

                                    indices.insert(indices.end(), k_indices.begin(), k_indices.end());
                                } // end num samples per probe
                            }
                        };

                        std::visit(visitor, generic_probe);
                    }
                }

                // TODO set flags
                auto readFlags = m_readFlagsSlot.CallAs<core_gl::FlagCallRead_GL>();
                auto writeFlags = m_writeFlagsSlot.CallAs<core_gl::FlagCallWrite_GL>();

                if (readFlags != nullptr && writeFlags != nullptr) {
                    (*readFlags)(core_gl::FlagCallWrite_GL::CallGetData);

                    if (readFlags->hasUpdate()) {
                        this->m_version = readFlags->version();
                    }

                    ++m_version;

                    auto flag_data = readFlags->getData();
                    auto kdtree_ids =
                        std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, indices, GL_DYNAMIC_DRAW);

                    if (!indices.empty()) {
                        m_filterAll_prgm->Enable();

                        auto flag_cnt = static_cast<GLuint>(flag_data->flags->getByteSize() / sizeof(GLuint));

                        glUniform1ui(m_filterAll_prgm->ParameterLocation("flag_cnt"), flag_cnt);

                        flag_data->flags->bind(1);

                        m_filterAll_prgm->Dispatch(static_cast<int>(std::ceil(flag_cnt / 64.0f)), 1, 1);

                        ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                        m_filterAll_prgm->Disable();

                        m_setFlags_prgm->Enable();

                        glUniform1ui(m_setFlags_prgm->ParameterLocation("id_cnt"), static_cast<GLuint>(indices.size()));

                        kdtree_ids->bind(0);
                        flag_data->flags->bind(1);

                        m_setFlags_prgm->Dispatch(static_cast<int>(std::ceil(indices.size() / 64.0f)), 1, 1);

                        ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                        m_setFlags_prgm->Disable();
                    }

                    writeFlags->setData(readFlags->getData(), m_version);
                    (*writeFlags)(core_gl::FlagCallWrite_GL::CallGetData);
                }
            }
        }


        // process probe selection events
        {
            auto pending_filter_event = event_collection->get<DataFilterByProbingDepth>();

            if (!pending_filter_event.empty()) {

                // TODO get corresponding data points from kd-tree
                auto tree = ct->getData();
                std::vector<uint32_t> indices;

                for (size_t probe_idx = 0; probe_idx < m_probe_selection.size(); ++probe_idx) {

                    auto generic_probe = probes->getGenericProbe(probe_idx);

                    auto visitor = [&tree, &indices, &pending_filter_event](auto&& arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, probe::Vec4Probe> || std::is_same_v<T, probe::FloatProbe>) {
                            auto position = arg.m_position;
                            auto direction = arg.m_direction;
                            auto begin = arg.m_begin;
                            auto end = arg.m_end;
                            auto samples_per_probe = arg.getSamplingResult()->samples.size();

                            auto sample_step = end / static_cast<float>(samples_per_probe);
                            auto radius = sample_step * 2.0; // sample_radius_factor;

                            float depth = std::min(end, pending_filter_event.back().depth);
                            //float depth = pending_filter_event.back().depth;

                            pcl::PointXYZ sample_point;
                            sample_point.x = position[0] + depth * direction[0];
                            sample_point.y = position[1] + depth * direction[1];
                            sample_point.z = position[2] + depth * direction[2];

                            std::vector<float> k_distances;
                            std::vector<uint32_t> k_indices;

                            auto num_neighbors =
                                tree->radiusSearch(sample_point, arg.m_sample_radius, k_indices, k_distances);
                            if (num_neighbors == 0) {
                                num_neighbors = tree->nearestKSearch(sample_point, 1, k_indices, k_distances);
                            }

                            indices.insert(indices.end(), k_indices.begin(), k_indices.end());
                        }
                    };

                    std::visit(visitor, generic_probe);
                }

                // TODO set flags
                auto readFlags = m_readFlagsSlot.CallAs<core_gl::FlagCallRead_GL>();
                auto writeFlags = m_writeFlagsSlot.CallAs<core_gl::FlagCallWrite_GL>();

                if (readFlags != nullptr && writeFlags != nullptr) {
                    (*readFlags)(core_gl::FlagCallWrite_GL::CallGetData);

                    if (readFlags->hasUpdate()) {
                        this->m_version = readFlags->version();
                    }

                    ++m_version;

                    auto flag_data = readFlags->getData();
                    auto kdtree_ids =
                        std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, indices, GL_DYNAMIC_DRAW);

                    if (!indices.empty()) {
                        m_filterAll_prgm->Enable();

                        auto flag_cnt = static_cast<GLuint>(flag_data->flags->getByteSize() / sizeof(GLuint));

                        glUniform1ui(m_filterAll_prgm->ParameterLocation("flag_cnt"), flag_cnt);

                        flag_data->flags->bind(1);

                        m_filterAll_prgm->Dispatch(static_cast<int>(std::ceil(flag_cnt / 64.0f)), 1, 1);

                        ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                        m_filterAll_prgm->Disable();

                        m_setFlags_prgm->Enable();

                        glUniform1ui(m_setFlags_prgm->ParameterLocation("id_cnt"), static_cast<GLuint>(indices.size()));

                        kdtree_ids->bind(0);
                        flag_data->flags->bind(1);

                        m_setFlags_prgm->Dispatch(static_cast<int>(std::ceil(indices.size() / 64.0f)), 1, 1);

                        ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                        m_setFlags_prgm->Disable();
                    }

                    writeFlags->setData(readFlags->getData(), m_version);
                    (*writeFlags)(core_gl::FlagCallWrite_GL::CallGetData);
                }
            }
        }
    }

    return true;
}

void FilterByProbe::PreRender(core_gl::view::CallRender3DGL& call) {}

} // namespace probe_gl
} // namespace megamol
