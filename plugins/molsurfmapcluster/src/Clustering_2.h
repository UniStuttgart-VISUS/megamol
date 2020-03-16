#pragma once

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"

namespace megamol {
namespace molsurfmapcluster {

class Clustering_2 : public core::Module {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "Clustering_2"; }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "Clusters Molecular Surface Maps"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Constructor */
    Clustering_2(void);

    /** Destructor */
    virtual ~Clustering_2(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool GetDataCallback(core::Call& caller);

    /**
     * Gets the data extents from the source.
     *
     * @param caller The calling call.
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool GetExtentCallback(core::Call& caller);

private:
    /** Slot to retrieve the image data */
    core::CallerSlot getImageSlot;

    /** Slot to send the resulting clustering over */
    core::CalleeSlot sendClusterSlot;

    /** The last hash of the incoming data */
    uint64_t lastDataHash;

    /** Offset to the incoming hash */
    uint64_t dataHashOffset;
};

} // namespace MolSurfMapClustering
} // namespace megamol
