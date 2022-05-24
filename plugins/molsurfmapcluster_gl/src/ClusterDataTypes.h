#pragma once

#include "vislib/graphics/BitmapImage.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace megamol {
namespace molsurfmapcluster_gl {

struct ClusterNode_2 {
    /** The ID of this node */
    int64_t id = -1;

    /** The ID of the left child. -1 if no left child exists */
    int64_t left = -1;

    /** The ID of the right child. -1 if no right child exists */
    int64_t right = -1;

    /** The ID of the parent node. -1 if no parent node exists */
    int64_t parent = -1;

    /** The ID of the representative node. -1 or id, if this node is the representative */
    int64_t representative = -1;

    /** The number of leaf nodes this node has. If this node is a leaf node, the value will be 1. */
    int64_t numLeafNodes = 1;

    /** The PDB id of the represented protein */
    std::string pdbID = "";

    /** Path to the represented picture file*/
    std::string picturePath = "";

    /** Path to the represented value file */
    std::string valueImagePath = "";

    /** Pointer to the represented image */
    std::weak_ptr<vislib::graphics::BitmapImage> picturePtr;

    /** Map that stores the distance between this node and all other nodes */
    std::map<int64_t, float> distanceMap;

    /** The height of the node in the hierarchy */
    float height = 0.0f;
};

/** Enum representing all available clustering methods */
enum class ClusteringMethod { IMAGE_MOMENTS = 0, COLOR_MOMENTS = 1, FILEFEATURES = 2 };

/** Enum representing all available distance measures */
enum class DistanceMeasure {
    EUCLIDEAN_DISTANCE = 0,
    L3_DISTANCE = 1,
    COSINUS_DISTANCE = 2,
    DICE_DISTANCE = 3,
    JACCARD_DISTANCE = 4
};

/** Struct representing a feature vector, possibly read from file */
struct FeatureData {
    std::string pdb_id;
    std::vector<float> feature_vec;
};

} // namespace molsurfmapcluster_gl
} // namespace megamol
