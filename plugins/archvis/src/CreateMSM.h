/*
 * CreateMSM.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef CREATE_MSM_H_INCLUDED
#define CREATE_MSM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "MSMDataCall.h"
#include "archvis/archvis.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"

namespace megamol {
namespace archvis {

class CreateMSM : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "CreateMSM"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Create Maﬂstabsmodell from float table input."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    CreateMSM();
    ~CreateMSM();

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
    std::shared_ptr<ScaleModel> m_MSM;

    /** The slot for requesting data */
    megamol::core::CalleeSlot m_getData_slot;

    /** The data callee slot. */
    megamol::core::CallerSlot m_node_floatTable_slot;

    /** The data callee slot. */
    megamol::core::CallerSlot m_element_floatTable_slot;

    /** The data callee slot. */
    megamol::core::CallerSlot m_inputElement_floatTable_slot;

    /** The data callee slot. */
    megamol::core::CallerSlot m_displacement_floatTable_slot;

    // TODO additional inputs?

    uint64_t m_node_input_hash;
    uint64_t m_element_input_hash;
    uint64_t m_inputElement_input_hash;
    uint64_t m_displacement_input_hash;

    uint64_t m_my_hash;
};

} // namespace archvis
} // namespace megamol

#endif // !CREATE_MSM_H_INCLUDED
