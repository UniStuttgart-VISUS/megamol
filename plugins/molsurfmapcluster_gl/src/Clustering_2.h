#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "ClusterDataTypes.h"

#include <array>

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

    /**
     * Checks whether at least one Parameter is dirty and resets their dirty flags
     *
     * @return True, if at least one parameter was dirty, false otherwise
     */
    bool checkParameterDirtyness(void);

    bool runComputation(void);

    bool clusterImages(void);

    bool calculateFeatureVectors(void);

    /** Slot to retrieve the image data */
    core::CallerSlot getImageSlot;
    core::CallerSlot getImageSlot2;
    core::CallerSlot getImageSlot3;
    core::CallerSlot getImageSlot4;

    /** Slot to send the resulting clustering over */
    core::CalleeSlot sendClusterSlot;

    /** Parameter to enable clustering using multiple maps */
    core::param::ParamSlot useMultipleMapsParam;

    /** Parameter to select the distance measure */
    core::param::ParamSlot distanceMeasureSelectionParam;

    /** Parameter to select the clustering method */
    core::param::ParamSlot clusteringMethodSelectionParam;

    /** Pointer to the vector containing the cluster nodes */
    std::shared_ptr<std::vector<ClusterNode_2>> nodes;

    /** Bool indicating whether the clustering has to be recalculated */
    bool recalculateClustering;

    /** The last hash of the incoming data */
    std::array<uint64_t, 4> lastDataHash;

    /** Offset to the incoming hash */
    uint64_t dataHashOffset;
};

} // namespace molsurfmapcluster
} // namespace megamol
