#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib_gl/graphics/gl/GLSLComputeShader.h"
#include "vislib_gl/graphics/gl/ShaderSource.h"

#include "probe/ProbeCalls.h"

#include "datatools/table/TableDataCall.h"

namespace megamol::probe_gl {
class ComputeDistance : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ComputeDistance";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "...";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    ComputeDistance();

    virtual ~ComputeDistance();

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    bool get_extent_cb(core::Call& c);

    core::CalleeSlot _out_table_slot;

    core::CallerSlot _in_probes_slot;

    core::param::ParamSlot _stretching_factor_slot;

    std::size_t _row_count = 0;

    std::size_t _col_count = 0;

    std::vector<datatools::table::TableDataCall::ColumnInfo> _col_infos;

    std::vector<float> _dis_mat;

    unsigned int _frame_id = std::numeric_limits<unsigned int>::max();

    std::size_t _out_data_hash = 0;

    vislib_gl::graphics::gl::GLSLComputeShader _fd_shader;

    vislib_gl::graphics::gl::ShaderSource _compute_shader_src;
};
} // namespace megamol::probe_gl
