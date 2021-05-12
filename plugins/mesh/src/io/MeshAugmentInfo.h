#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "mesh/MeshCalls.h"

#include "mmstd_datatools/table/TableDataCall.h"

namespace megamol::mesh::io {

class MeshAugmentInfo : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "MeshAugmentInfo";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "MeshAugmentInfo";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    MeshAugmentInfo();

    virtual ~MeshAugmentInfo();

protected:
    bool create() override;

    void release() override;

private:
    bool is_dirty() const {
        return filename_slot_.IsDirty();
    }

    void reset_dirty() {
        filename_slot_.ResetDirty();
    }

    bool get_mesh_data_cb(core::Call& c);

    bool get_mesh_extent_cb(core::Call& c);

    bool get_info_data_cb(core::Call& c);

    bool get_info_extent_cb(core::Call& c);

    bool assert_data(CallMesh& mesh, core::view::CallGetTransferFunction& tf);

    core::CalleeSlot mesh_out_slot_;

    core::CalleeSlot info_out_slot_;

    core::CallerSlot mesh_in_slot_;

    core::CallerSlot tf_in_slot_;

    core::param::ParamSlot filename_slot_;

    std::vector<float> infos_;

    std::vector<float> colors_;

    std::shared_ptr<MeshDataAccessCollection> meshes_;

    stdplugin::datatools::table::TableDataCall::ColumnInfo column_info_;

    std::size_t out_data_hash_ = 0;
};

} // namespace megamol::mesh::io
