#pragma once

#include <optional>

#include "mmcore/CallerSlot.h"
#include "mmcore/UniFlagCalls.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer3DModuleGL.h"

#include "mesh/MeshCalls.h"

#include "mmstd_datatools/StatisticsCall.h"

#include "implot.h"

namespace megamol::thermodyn {
class MeshWidget : public core::view::Renderer3DModuleGL {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "MeshWidget";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "MeshWidget";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    MeshWidget();

    virtual ~MeshWidget();

protected:
    bool create() override;

    void release() override;

private:
    bool Render(core::view::CallRender3DGL& call) override;

    bool GetExtents(core::view::CallRender3DGL& call) override;

    bool widget(float x, float y, std::size_t idx, mesh::MeshDataAccessCollection::Mesh const& mesh,
        core::moldyn::SimpleSphericalParticles const* info,
        std::vector<stdplugin::datatools::StatisticsData> const& stats);

    bool widget(float x, float y,
        std::list<std::pair<std::size_t, mesh::MeshDataAccessCollection::Mesh const*>> const& selected,
        core::moldyn::SimpleSphericalParticles const* info,
        std::vector<stdplugin::datatools::StatisticsData> const& stats);

    bool parse_data(mesh::CallMesh& in_mesh, core::moldyn::MultiParticleDataCall* in_info,
        stdplugin::datatools::StatisticsCall* in_stats, core::FlagCallRead_CPU& fcr);

    bool OnMouseMove(double x, double y) override;

    core::CallerSlot in_data_slot_;

    core::CallerSlot in_info_slot_;

    core::CallerSlot in_stats_slot_;

    core::CallerSlot flags_read_slot_;

    core::param::ParamSlot accumulate_slot_;

    ImPlotContext* ctx_ = nullptr;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    int frame_id_ = -1;

    float mouse_x_;
    float mouse_y_;

    std::vector<std::size_t> mesh_prefix_count_;
};
} // namespace megamol::thermodyn
