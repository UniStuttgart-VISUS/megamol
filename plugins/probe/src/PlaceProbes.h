/*
 * PlaceProbes.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef PLACE_PROBES_H_INCLUDED
#define PLACE_PROBES_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "ProbeCollection.h"

namespace megamol {
namespace probe {

class PlaceProbes : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() { return "PlaceProbes"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() { return "..."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    PlaceProbes();

    /** Dtor. */
    virtual ~PlaceProbes();

protected:
    virtual bool create();
    virtual void release();

    core::CallerSlot m_mesh_call;
    size_t           m_mesh_call_cached_hash;

    core::CalleeSlot m_probe_call;
    size_t           m_probe_call_cached_hash;
    
private:
    bool getData(core::Call& call);

    bool getMetaData(core::Call& call);

    std::shared_ptr<ProbeCollection> m_probes;
};


} // namespace probe
} // namespace megamol

#endif //!PLACE_PROBES_H_INCLUDED