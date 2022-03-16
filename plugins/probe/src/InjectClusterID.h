/*
 * InjectClusterID.h
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef INJECT_CLUSTER_ID_H_INCLUDED
#define INJECT_CLUSTER_ID_H_INCLUDED

#include "mesh/MeshCalls.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

namespace megamol {
namespace probe {

class InjectClusterID : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "InjectClusterID";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Injects the cluster ID of a probe into the hull mesh (swaps probe ID with cluster ID)";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    InjectClusterID(void);

    /** Dtor. */
    virtual ~InjectClusterID(void);

protected:
    virtual bool create();
    virtual void release();

    core::CallerSlot _probes_rhs_slot;
    core::CallerSlot _mesh_rhs_slot;
    core::CalleeSlot _mesh_lhs_slot;

private:
    bool getMetaData(core::Call& call);
    bool getData(core::Call& call);

    uint32_t _version = 0;
};

} // namespace probe
} // namespace megamol

#endif // !TESSELLATE_BOUNDING_BOX_H_INCLUDED
