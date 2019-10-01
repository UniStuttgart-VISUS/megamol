/*
 * SampleAlongProbes.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */


#ifndef SAMPLE_ALONG_PROBES_H_INCLUDED
#define SAMPLE_ALONG_PROBES_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "ProbeCollection.h"

namespace megamol {
namespace probe {

class SampleAlongPobes : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() { return "SampleAlongProbes"; }

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

    SampleAlongPobes();
    virtual ~SampleAlongPobes();

protected:
    virtual bool create();
    virtual void release();

    core::CalleeSlot m_probe_lhs_slot;
    size_t m_probe_lhs_cached_hash;

    core::CallerSlot m_probe_rhs_slot;
    size_t m_probe_rhs_cached_hash;

    core::CallerSlot m_adios_rhs_slot;
    size_t m_adios_rhs_cached_hash;

private:
    bool getData(core::Call& call);

    bool getMetaData(core::Call& call);

    std::shared_ptr<ProbeCollection> m_probes;
};

} // namespace probe
} // namespace megamol


#endif // !SAMPLE_ALONG_PROBES_H_INCLUDED
