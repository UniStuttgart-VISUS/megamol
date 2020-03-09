/*
 * HierarchicalClustering.h
 *
 * Copyright (C) 2019 by Tobias Baur
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MOLSURFMAPCLUSTER_HIERARCHICALCLUSTERING_INCLUDED
#define MOLSURFMAPCLUSTER_HIERARCHICALCLUSTERING_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <tuple>
#include <vector>

#include "PictureData.h"

namespace megamol {
namespace MolSurfMapCluster {


class HierarchicalClustering {
public:
    struct CLUSTERNODE {
        // ID of Node
        int id;
        int level;

        // Beziehungen
        CLUSTERNODE* parent;
        CLUSTERNODE* clusterparent;
        CLUSTERNODE* left;
        CLUSTERNODE* right;

        double similiaritychildren;

        // Picture Data
        PictureData* pic;

        // Featrues vector
        std::vector<double>* features;

        // Distance Matrix
        std::vector<std::tuple<CLUSTERNODE*, double>>* distances;

        // PCA
        std::vector<double>* pca2d;
    };

    /** Constructor */
    HierarchicalClustering(void);
    HierarchicalClustering(std::vector<PictureData>&, SIZE_T, bool = true, int = 2, int = 1, int = 1,
        int = 1); // Standard Euclidian Distance, Avarage Linkage, Imge-Moments
    HierarchicalClustering(
        HierarchicalClustering::CLUSTERNODE*, SIZE_T, bool = true, int = 2, int = 1, int = 1, int = 1);

    /** Destructor */
    virtual ~HierarchicalClustering(void);

    bool finished();

    void calculateImageMoments(CLUSTERNODE*);
    void calculateImageMomentsValue(CLUSTERNODE*);
    void calculateColorMoments(CLUSTERNODE*);
    void calculateColorMomentsValue(CLUSTERNODE*);
    std::vector<double>* gray_scale_image(PictureData*);

    void dump_dot();
    void dump_dot(const vislib::TString&);
    void dump_pca(const vislib::TString&);

    void clusterthedata(void);
    void reanalyse(bool actualValue = true);
    void setDistanceMethod(unsigned int);
    void setSimilarityMethod(unsigned int);
    void setLinkageMethod(unsigned int);
    void setDistanceMultiplier(float);
    void changeModeTo(short int);
    void setMoments(short int);

    std::vector<HierarchicalClustering::CLUSTERNODE*>* getAllNodes(void) { return this->cluster; }
    bool parentIs(HierarchicalClustering::CLUSTERNODE*, HierarchicalClustering::CLUSTERNODE*);
    std::vector<CLUSTERNODE*>* getLeaves(void);
    std::vector<HierarchicalClustering::CLUSTERNODE*>* getLeavesOfNode(HierarchicalClustering::CLUSTERNODE*);
    CLUSTERNODE* getRoot(void);
    std::vector<CLUSTERNODE*>* getClusterNodesOfNode(CLUSTERNODE*);
    CLUSTERNODE* getClusterRootOfNode(CLUSTERNODE* node);
    double distance(std::vector<double>*, std::vector<double>*, int = 2);

    double getMaxDistanceOfLeavesToRoot();

private:
    std::vector<CLUSTERNODE*>* cluster;
    std::vector<CLUSTERNODE*>* root;
    std::vector<CLUSTERNODE*>* leaves;

    double maxdistance;

    int distancemethod;
    int similaritymethod;
    int linkagemethod;
    int momentsmethode;
    float distancemultiplier = 0.75;

    short int mode;

    bool clusteringfinished;

    std::string getFeaturesString(std::vector<double>*);

    double similarity(std::vector<double>*, std::vector<double>*, int = 1);

    CLUSTERNODE* mergeCluster(CLUSTERNODE* cluster1, CLUSTERNODE* cluster2);

    PictureData* findNearestPicture(std::vector<double>* features, std::vector<HierarchicalClustering::CLUSTERNODE*>* leftleaves,
        std::vector<HierarchicalClustering::CLUSTERNODE*>* rightleaves);

    HierarchicalClustering::CLUSTERNODE* getRootOfNode(HierarchicalClustering::CLUSTERNODE*);

    void calculatePCA(std::vector<HierarchicalClustering::CLUSTERNODE*>*);

    static uint32_t bla;
};

} // namespace MolSurfMapCluster
} // namespace megamol

#endif
