#pragma once

#include "CallClustering_2.h"
#include "ClusterDataTypes.h"
#include "image_calls/Image2DCall.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include <array>
#include <filesystem>

namespace megamol {
namespace molsurfmapcluster_gl {

class Clustering_2 : public core::Module {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "Clustering_2";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Clusters Molecular Surface Maps";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

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

    bool calculateFileFeatureVectors(std::vector<std::pair<image_calls::Image2DCall*, int>> const& calls);

    bool calculateMomentsFeatureVectors(
        std::vector<std::pair<image_calls::Image2DCall*, int>> const& calls, ClusteringMethod method);

    std::string valumeImageNameFromNormalImage(const std::string& str) const;

    bool loadFeatureVectorFromFile(
        const std::filesystem::path& file_path, std::map<std::string, std::vector<float>>& OUT_feature_map) const;

    bool loadValueImageFromFile(const std::filesystem::path& file_path, std::vector<float>& OUT_value_image,
        bool normalize_values = true) const;

    void calcColorMomentsForValueImage(
        std::vector<float> const& val_image, std::vector<float>& OUT_feature_vector) const;

    void calcImageMomentsForValueImage(
        std::vector<float> const& val_image, std::vector<float>& OUT_feature_vector) const;

    void calcFeatureVectorForNonLeaf(std::vector<int64_t> const& leaf_node_ids, std::vector<float>& OUT_feature_vector) const;

    float calcNodeNodeDistance(ClusterNode_2 const& node1, ClusterNode_2 const& node2) const;

    float calcFeatureVectorDistance(std::vector<float> const& vec1, std::vector<float> const& vec2) const;

    void mergeClusters(int64_t n_1, int64_t n_2, std::vector<int64_t> const& root_vec);

    void postprocessClustering();

    std::vector<int64_t> calcLeaveIndicesOfNode(int64_t node_id) const;

    std::filesystem::path getPathForIndex(int const index);

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

    core::param::ParamSlot image_1_features_slot_;
    core::param::ParamSlot image_2_features_slot_;
    core::param::ParamSlot image_3_features_slot_;
    core::param::ParamSlot image_4_features_slot_;

    /** Pointer to the vector containing the cluster nodes */
    ClusteringData node_data_;

    std::map<std::string, std::vector<float>> feature_vectors_;

    /** Bool indicating whether the clustering has to be recalculated */
    bool recalculateClustering;

    /** The last hash of the incoming data */
    std::array<uint64_t, 4> lastDataHash;

    /** Offset to the incoming hash */
    uint64_t dataHashOffset;

    /** The standard width for the input pictures */
    static constexpr int picwidth = 2560;

    /** The standard height for the input pictures */
    static constexpr int picheight = 1440;
};

} // namespace molsurfmapcluster_gl
} // namespace megamol
