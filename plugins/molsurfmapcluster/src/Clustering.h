/*
 * Clustering.h
 *
 * Copyright (C) 2019 by Tobias Baur
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MOLSURFMAPCLUSTER_CLUSTERING_INCLUDED
#define MOLSURFMAPCLUSTER_CLUSTERING_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <vector>
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "png.h"

#include "vislib/String.h"
#include "vislib/math/Cuboid.h"

#include <filesystem>

#include "CallClusteringLoader.h"
#include "CallPNGPics.h"
#include "HierarchicalClustering.h"

namespace megamol {
namespace molsurfmapcluster {

/**
 * Loader module for simple ASCII comma seperated files
 * which contain data for single spheres for each line.
 */
class Clustering : public core::Module {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * TUTORIAL: Mandatory method for every module or call that states the name of the class.
     * This name should be unique.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "Clustering"; }

    /**
     * Gets a human readable description of the module.
     *
     * TUTORIAL: Mandatory method for every module or call that returns a description.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "Clusters PNG-Pictures"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * TUTORIAL: Mandatory method for every module.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Constructor */
    Clustering(void);

    /** Destructor */
    virtual ~Clustering(void);

private:
    /**
     * Ensures that the data is loaded
     */
    void clusterData(image_calls::Image2DCall*, image_calls::Image2DCall* = nullptr, image_calls::Image2DCall* = nullptr);
    void clusterData(CallClusteringLoader*);

    /**
     * Implementation of 'Create'.
     *
     * TUTORIAL: The overwritten version of this method gets called right after an object of this class has been
     *instantiated.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Gets the data from the source.
     *
     * TUTORIAL: This method computes the final data and writes them to the calling call.
     * Data that was written by getExtentCallback should be written again, just in case...
     *
     * @param caller The calling call.
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getDataCallback(core::Call& caller);

    /**
     * Gets the data extents from the source.
     *
     * TUTORIAL: This method computes the extents of the data set, namely the bounding box and other relevant values,
     * and writes them into the calling call.
     *
     * @param caller The calling call.
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getExtentCallback(core::Call& caller);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    void fillPictureDataVector(image_calls::Image2DCall& imc);

    void loadValueImageForGivenPicture(const std::filesystem::path& originalPicture, std::vector<float>& outValueImage); 

    bool loadFeatureVectorFromFile(const std::filesystem::path& filePath, std::map<std::string, std::vector<float>>& outFeatureMap);

    /** The slot for requesting data. */
    core::CalleeSlot outSlot;

    /** The slot for getting data. */
    core::CallerSlot inSlotImageLoader;
    core::CallerSlot inSlotImageLoader2;
    core::CallerSlot inSlotImageLoader3;
    core::CallerSlot inSlotCLUSTERINGLoader;

    /** The data update hash */
    SIZE_T lastHash;
    SIZE_T outHash;
    SIZE_T hashoffset;
    SIZE_T outhashoffset;

    bool datatocluster;

    SIZE_T picturecount;

    // Cluster Datastructure
    HierarchicalClustering* clustering;

    /** The bounding box */
    vislib::math::Cuboid<float> bbox;

    std::vector<PictureData> picdata;

    /** Param Slots */
    core::param::ParamSlot dumpdot;
    core::param::ParamSlot dumpdotpath;
    core::param::ParamSlot distanceFilePath;
    core::param::ParamSlot selectionmode;
    core::param::ParamSlot linkagemodeparam;
    core::param::ParamSlot distancemultiplier;
    core::param::ParamSlot momentsmethode;
    core::param::ParamSlot featuresSlot1Param;
    core::param::ParamSlot featuresSlot2Param;
    core::param::ParamSlot featuresSlot3Param;
    bool dumpdotfile;
    bool selectionmodechanged;
    bool linkagemodechanged;
    bool distancemultiplierchanged;
    bool momentschanged;

    std::map<std::string, std::vector<float>> slot1Features;
    std::map<std::string, std::vector<float>> slot2Features;
    std::map<std::string, std::vector<float>> slot3Features;

    std::map<std::string, std::map<std::string, double>> distanceMatrix;
};

enum ClusteringMode {
    COSINUS = 1,
    DICE = COSINUS + 1,
    JACCARD = DICE + 1,
    OVERLAP = JACCARD + 1,
    CITYBLOCK = OVERLAP + 1,
    EUCLIDIAN = CITYBLOCK + 1,
    L3 = EUCLIDIAN + 1,
    GOWER = L3 + 1
};

enum LinkageMode { CENTROIDE = 1, SINGLE = CENTROIDE + 1, AVARAGE = SINGLE + 1 };

enum MomentsMethode { IMAGE = 1, COLOR = IMAGE + 1, AIFEATURES = COLOR + 1 };

} // namespace MolSurfMapCluster
} /* end namespace megamol */

#endif /*  MOLSURFMAPCLUSTER_CLUSTERING_INCLUDED */
