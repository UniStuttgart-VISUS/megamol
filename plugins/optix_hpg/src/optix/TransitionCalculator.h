#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "mesh/MeshCalls.h"

#include "cuda.h"
#include "optix.h"

#include "optix/Utils.h"

#include "MMOptixModule.h"
#include "SBT.h"

#include "optix/Context.h"

namespace megamol::optix_hpg {
class TransitionCalculator : public core::Module {
public:
    std::vector<std::string> requested_lifetime_resources() override {
        auto res = Module::requested_lifetime_resources();
        res.push_back(frontend_resources::CUDA_Context_Req_Name);
        return res;
    }

    static const char* ClassName(void) {
        return "TransitionCalculator";
    }

    static const char* Description(void) {
        return "TransitionCalculator for OptiX";
    }

    static bool IsAvailable(void) {
        return true;
    }

    TransitionCalculator();

    virtual ~TransitionCalculator();

protected:
    bool create() override;

    void release() override;

private:
    enum class output_type : std::uint8_t { outbound, inbound };

    bool init();

    bool assertData(mesh::CallMesh& mesh, core::moldyn::MultiParticleDataCall& particles,
        core::view::CallGetTransferFunction& tf, unsigned int frameID);

    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    core::CalleeSlot out_transitions_slot_;

    core::CallerSlot in_mesh_slot_;

    core::CallerSlot in_paths_slot_;

    core::CallerSlot in_tf_slot_;

    core::param::ParamSlot output_type_slot_;

    core::param::ParamSlot frame_count_slot_;

    core::param::ParamSlot frame_skip_slot_;

    OptixModule builtin_triangle_intersector_;

    MMOptixModule mesh_module_;

    MMOptixModule raygen_module_;

    MMOptixModule miss_module_;

    std::vector<CUdeviceptr> ray_buffer_;

    OptixPipeline pipeline_;

    MMOptixSBT sbt_;

    unsigned int _frame_id = std::numeric_limits<unsigned int>::max();

    std::size_t _data_hash = std::numeric_limits<std::size_t>::max();

    std::unique_ptr<Context> optix_ctx_;

    std::vector<std::vector<glm::vec4>> colors_;

    std::vector<std::vector<glm::vec3>> positions_;

    std::vector<std::vector<std::uint32_t>> indices_;

    std::shared_ptr<mesh::MeshDataAccessCollection> mesh_access_collection_;

    std::size_t out_data_hash_ = 0;

    std::vector<std::vector<uint64_t>> ray_vec_ident_;

    std::vector<std::vector<uint8_t>> ray_vec_active_;
};
} // namespace megamol::optix_hpg
