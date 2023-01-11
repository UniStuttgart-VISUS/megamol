#pragma once

#include "FrameStatistics.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "geometry_calls/MultiParticleDataCall.h"

namespace megamol {
namespace datatools_gl {
namespace misc {

/**
 * This module shows a table for debugging MultiParticleDataCall.
 */
class ParticleInspector : public megamol::core::Module {

public:
    void requested_lifetime_resources(frontend_resources::ResourceRequest& req) override {
        Module::requested_lifetime_resources(req);
        req.require<frontend_resources::FrameStatistics>();
    }

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName(void) {
        return "ParticleInspector";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description(void) {
        return "Direct inspection of particle values, data is passed through.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable(void) {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    ParticleInspector(void);

    /**
     * Finalises an instance.
     */
    virtual ~ParticleInspector(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    bool getParticleData(core::Call& call);

    bool getParticleExtents(core::Call& call);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

private:
    void drawTable(geocalls::MultiParticleDataCall* c);

    /** The slot for retrieving the data as multi particle data. */
    core::CalleeSlot slotParticlesOut;

    /** The data callee slot. */
    core::CallerSlot slotParticlesIn;

    uint32_t lastDrawnFrame = std::numeric_limits<uint32_t>::max();
};

} /* end namespace misc */
} // namespace datatools_gl
} /* end namespace megamol */
