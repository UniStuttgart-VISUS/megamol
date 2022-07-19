/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "ArchVisCalls.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol::archvis_gl {

class CreateFEMModel : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "CreateFEMModel";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Create FEM model from float table input.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    CreateFEMModel();
    ~CreateFEMModel();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& caller);

    /**
     * Implementation of 'Release'.
     */
    void release();

private:
    uint32_t m_version;

    std::shared_ptr<FEMModel> m_FEM_model;

    /** First FEM parameter */
    core::param::ParamSlot m_fem_param_0;

    /** Second FEM parameter */
    core::param::ParamSlot m_fem_param_1;

    /** The slot for requesting data */
    megamol::core::CalleeSlot m_getData_slot;

    /** The data callee slot. */
    megamol::core::CallerSlot m_node_floatTable_slot;

    /** The data callee slot. */
    megamol::core::CallerSlot m_element_floatTable_slot;

    /** The data callee slot. */
    megamol::core::CallerSlot m_deformation_floatTable_slot;

    // TODO additional inputs?

    uint64_t m_node_input_hash;
    uint64_t m_element_input_hash;
    uint64_t m_deform_input_hash;
};

} // namespace megamol::archvis_gl
