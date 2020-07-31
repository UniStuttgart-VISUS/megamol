#include "FilterByProbe.h"

#include "CallKDTree.h"
#include "ProbeCalls.h"
#include "ProbeGlCalls.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/FlagCall_GL.h"

#include "ProbeCollection.h"

namespace megamol {
namespace probe_gl {

FilterByProbe::FilterByProbe()
    : m_version(0)
    , m_probes_slot("getProbes", "")
    , m_kd_tree_slot("getKDTree", "")
    , m_probe_manipulation_slot("getProbeInteraction", "")
    , m_readFlagsSlot("getReadFlags", "")
    , m_writeFlagsSlot("getWriteFlags", "") {

    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_kd_tree_slot.SetCompatibleCall<probe::CallKDTreeDescription>();
    this->MakeSlotAvailable(&this->m_kd_tree_slot);

    this->m_probe_manipulation_slot.SetCompatibleCall<CallProbeInteractionDescription>();
    this->MakeSlotAvailable(&this->m_probe_manipulation_slot);

    this->m_readFlagsSlot.SetCompatibleCall<core::FlagCallRead_GLDescription>();
    this->MakeSlotAvailable(&this->m_readFlagsSlot);

    this->m_writeFlagsSlot.SetCompatibleCall<core::FlagCallWrite_GLDescription>();
    this->MakeSlotAvailable(&this->m_writeFlagsSlot);
}

FilterByProbe::~FilterByProbe() { this->Release(); }

bool FilterByProbe::create() {
    try {
        // create shader program
        m_setFlags_prgm = std::make_unique<GLSLComputeShader>();
        m_filterAll_prgm = std::make_unique<GLSLComputeShader>();

        vislib::graphics::gl::ShaderSource setFlags_src;
        vislib::graphics::gl::ShaderSource filterAll_src;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("FilterByProbe::setFlags", setFlags_src)) return false;
        if (!m_setFlags_prgm->Compile(setFlags_src.Code(), setFlags_src.Count())) return false;
        if (!m_setFlags_prgm->Link()) return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("FilterByProbe::filterAll", filterAll_src))
            return false;
        if (!m_filterAll_prgm->Compile(filterAll_src.Code(), filterAll_src.Count())) return false;
        if (!m_filterAll_prgm->Link()) return false;

    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to compile shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
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

bool FilterByProbe::GetExtents(core::view::CallRender3D_2& call) { return true; }

bool FilterByProbe::Render(core::view::CallRender3D_2& call) {
    probe::CallProbes* pc = this->m_probes_slot.CallAs<probe::CallProbes>();
    if (pc == NULL) return false;
    if (!(*pc)(0)) return false;

    // query kd tree data
    auto ct = this->m_kd_tree_slot.CallAs<probe::CallKDTree>();
    if (ct == nullptr) return false;
    if (!(*ct)(0)) return false;

    // TODO get probe interaction
    // check for pending probe manipulations
    CallProbeInteraction* pic = this->m_probe_manipulation_slot.CallAs<CallProbeInteraction>();
    if (pic != NULL) {
        if (!(*pic)(0)) return false;

        if (pic->hasUpdate()) {
            auto interaction_collection = pic->getData();

            auto& pending_manips = interaction_collection->accessPendingManipulations();

            auto probes = pc->getData();

            for (auto itr = pending_manips.begin(); itr != pending_manips.end(); ++itr) {
                if (itr->type == HIGHLIGHT) {
                    auto manipulation = *itr;

                } else if (itr->type == DEHIGHLIGHT) {
                    auto manipulation = *itr;

                } else if (itr->type == SELECT) {
                    auto manipulation = *itr;

                    // TODO get probe
                    auto generic_probe = probes->getGenericProbe(manipulation.obj_id);

                    // TODO get corresponding data points from kd-tree
                    auto tree = ct->getData();
                    std::vector<uint32_t> indices;

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

                    // TODO set flags
                    auto readFlags = m_readFlagsSlot.CallAs<core::FlagCallRead_GL>();
                    auto writeFlags = m_writeFlagsSlot.CallAs<core::FlagCallWrite_GL>();

                    if (readFlags != nullptr && writeFlags != nullptr) {
                        (*readFlags)(core::FlagCallWrite_GL::CallGetData);

                        if (readFlags->hasUpdate()) {
                            this->m_version = readFlags->version();
                        }

                        ++m_version;

                        auto flag_data = readFlags->getData();
                        auto kdtree_ids =
                            std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, indices, GL_DYNAMIC_DRAW);

                        {
                            m_filterAll_prgm->Enable();

                            auto flag_cnt = static_cast<GLuint>(flag_data->flags->getByteSize() / sizeof(GLuint));

                            glUniform1ui(m_filterAll_prgm->ParameterLocation("flag_cnt"), flag_cnt);

                            flag_data->flags->bind(1);

                            m_filterAll_prgm->Dispatch(static_cast<int>(std::ceil(flag_cnt / 64.0f)), 1, 1);

                            ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                            m_filterAll_prgm->Disable();
                        }

                        {
                            m_setFlags_prgm->Enable();

                            glUniform1ui(
                                m_setFlags_prgm->ParameterLocation("id_cnt"), static_cast<GLuint>(indices.size()));

                            kdtree_ids->bind(0);
                            flag_data->flags->bind(1);

                            m_setFlags_prgm->Dispatch(static_cast<int>(std::ceil(indices.size() / 64.0f)), 1, 1);

                            ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                            m_setFlags_prgm->Disable();
                        }

                        writeFlags->setData(readFlags->getData(), m_version);
                        (*writeFlags)(core::FlagCallWrite_GL::CallGetData);
                    }


                } else {
                    //
                }
            }
        }
    }

    return true;
}

void FilterByProbe::PreRender(core::view::CallRender3D_2& call) {}

} // namespace probe_gl
} // namespace megamol