#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "datatools/table/TableDataCall.h"
#include "geometry_calls//EllipsoidalDataCall.h"
#include "geometry_calls//MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include <array>
#include <map>

namespace megamol {
namespace datatools {

/**
 * This module converts from a generic table to the MultiParticleDataCall.
 */
class ParticlesToTable : public megamol::core::Module {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "ParticlesToTable";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Converts particles to generic tables.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    ParticlesToTable();

    /**
     * Finalises an instance.
     */
    ~ParticlesToTable() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    bool getTableData(core::Call& call);

    bool getTableHash(core::Call& call);

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    bool assertMPDC(geocalls::MultiParticleDataCall* in, table::TableDataCall* tc);
    bool assertEPDC(geocalls::EllipsoidalParticleDataCall* c, table::TableDataCall* tc);

    /** The slot for retrieving the data as multi particle data. */
    core::CalleeSlot slotTableOut;

    /** The data callee slot. */
    core::CallerSlot slotParticlesIn;

    std::vector<float> everything;

    SIZE_T inHash = SIZE_MAX;
    unsigned int inFrameID = std::numeric_limits<unsigned int>::max();
    std::vector<table::TableDataCall::ColumnInfo> column_infos;
    uint32_t total_particles = 0;
};

} /* end namespace datatools */
} /* end namespace megamol */
