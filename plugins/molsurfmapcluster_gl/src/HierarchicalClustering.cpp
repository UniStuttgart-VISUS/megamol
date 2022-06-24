/*
 * HierarchicalClustering.cpp
 *
 * Copyright (C) 2019 by Tobias Baur
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "mmcore/utility/log/Log.h"

#include "vislib/Array.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/Vector.h"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <sstream>

#include "HierarchicalClustering.h"
#include "PictureData.h"

#define IMAGEORDER 3

#define DISTANCEMODE 1
#define SIMILARITYMODE 2

#define PCA 2

using namespace megamol;
using namespace megamol::molsurfmapcluster;

uint32_t HierarchicalClustering::bla = 0;

HierarchicalClustering::HierarchicalClustering() {}

HierarchicalClustering::~HierarchicalClustering() {}

HierarchicalClustering::HierarchicalClustering(std::vector<PictureData>& pics, SIZE_T picturecount,
    const std::map<std::string, std::vector<float>>& features,
    const std::map<std::string, std::map<std::string, double>>& distmatrix, int method, int mode, int linkage,
    int moments) {

    // Initalize Variables
    this->cluster = new std::vector<CLUSTERNODE*>();
    this->root = new std::vector<CLUSTERNODE*>();
    this->leaves = new std::vector<CLUSTERNODE*>();
    this->distancemethod = method;
    this->similaritymethod = method;
    this->mode = mode;
    this->linkagemethod = linkage;
    this->momentsmethode = moments;
    this->distanceMatrix = distmatrix;

    this->clusteringfinished = false;

    // Create Datastructure
    core::utility::log::Log::DefaultLog.WriteMsg(core::utility::log::Log::LEVEL_INFO, "Creating Clustering");
    for (int index = 0; index < picturecount; index++) {
        CLUSTERNODE* node = new CLUSTERNODE();
        // Set ID
        node->id = this->leaves->size() + 1;
        node->level = 0;
        node->height = 0.0f;
        node->features = new std::vector<double>();

        // Set Beziehungen
        node->parent = nullptr;
        node->left = nullptr;
        node->right = nullptr;

        node->similiaritychildren = 1.0;

        // Set Picture
        PictureData* tmppicture = &(pics[index]);
        node->pic = tmppicture;

        // Init dtsiance Matrix
        node->distances = new std::vector<std::tuple<CLUSTERNODE*, double>>();

        this->leaves->push_back(node);
    }

    this->cluster->insert(this->cluster->end(), this->leaves->begin(), this->leaves->end());

    // Getting Features of Pictures
    for (CLUSTERNODE* node : *this->cluster) {
        if (this->momentsmethode == 1)
            HierarchicalClustering::calculateImageMomentsValue(node);
        if (this->momentsmethode == 2)
            HierarchicalClustering::calculateColorMomentsValue(node);
        if (this->momentsmethode == 3) {
            if (features.count(node->pic->pdbid) > 0) {
                const std::vector<float>& feat = features.at(node->pic->pdbid);
                node->features->insert(node->features->end(), feat.begin(), feat.end());
            }
        }
    }

    // Cluster the Data
    this->clusterthedata();
}

HierarchicalClustering::HierarchicalClustering(std::vector<PictureData>& pictures,
    const std::map<std::string, std::vector<float>>& features1,
    const std::map<std::string, std::vector<float>>& features2,
    const std::map<std::string, std::vector<float>>& features3,
    const std::map<std::string, std::map<std::string, double>>& distmatrix, image_calls::Image2DCall* call1,
    image_calls::Image2DCall* call2, image_calls::Image2DCall* call3, int method, int mode, int linkage, int moments) {

    // Initalize Variables
    this->cluster = new std::vector<CLUSTERNODE*>();
    this->root = new std::vector<CLUSTERNODE*>();
    this->leaves = new std::vector<CLUSTERNODE*>();
    this->distancemethod = method;
    this->similaritymethod = method;
    this->mode = mode;
    this->linkagemethod = linkage;
    this->momentsmethode = moments;
    this->distanceMatrix = distmatrix;

    this->clusteringfinished = false;

    std::vector<std::pair<uint32_t, image_calls::Image2DCall*>> calls = {
        std::make_pair(2u, call3), std::make_pair(1u, call2), std::make_pair(0u, call1)};
    size_t piccount = std::numeric_limits<size_t>::max();

    bool first = true;

    for (const auto call : calls) {
        if (call.second == nullptr)
            continue;
        pictures.clear();
        pictures.shrink_to_fit(); // force to clear everything

        // the wishlist is a nullptr, so we want all pictures
        if (!(*call.second)(image_calls::Image2DCall::CallForSetWishlist)) {
            core::utility::log::Log::DefaultLog.WriteError("ImageLoader function call failed");
            return;
        }
        if (!(*call.second)(image_calls::Image2DCall::CallForWaitForData)) {
            core::utility::log::Log::DefaultLog.WriteError("ImageLoader function call failed");
            return;
        }
        if (!(*call.second)(image_calls::Image2DCall::CallForGetData)) {
            core::utility::log::Log::DefaultLog.WriteError("ImageLoader function call failed");
            return;
        }
        if (!(*call.second)(image_calls::Image2DCall::CallForWaitForData)) {
            core::utility::log::Log::DefaultLog.WriteError("ImageLoader function call failed");
            return;
        }

        // load the images into the vector
        auto imcount = call.second->GetImagePtr()->size();
        pictures.resize(imcount);
        uint32_t id = 0;
        float globalmin = std::numeric_limits<float>::max();
        float globalmax = std::numeric_limits<float>::lowest();

        for (auto& p : *call.second->GetImagePtr()) {
            pictures[id].width = p.second.Width();
            pictures[id].height = p.second.Height();
            pictures[id].path = p.first;
            pictures[id].pdbid = std::filesystem::path(p.first).stem().string();
            pictures[id].render = false;
            pictures[id].popup = false;
            pictures[id].texture = nullptr;
            pictures[id].image = &p.second;
            auto minmax = this->loadValueImage(std::filesystem::path(p.first), pictures[id].valueImage);
            pictures[id].minValue = minmax.first;
            pictures[id].maxValue = minmax.second;

            if (minmax.first < globalmin)
                globalmin = minmax.first;
            if (minmax.second > globalmax)
                globalmax = minmax.second;

            if (first) {
                CLUSTERNODE* node = new CLUSTERNODE();
                // Set ID
                node->id = this->leaves->size() + 1;
                node->level = 0;
                node->height = 0.0f;
                node->features = new std::vector<double>();

                // Set Beziehungen
                node->parent = nullptr;
                node->left = nullptr;
                node->right = nullptr;

                node->similiaritychildren = 1.0;

                // Set Picture
                PictureData* tmppicture = &(pictures[id]);
                node->pic = tmppicture;

                // Init dtsiance Matrix
                node->distances = new std::vector<std::tuple<CLUSTERNODE*, double>>();

                this->leaves->push_back(node);
            } else {
                leaves->at(id)->pic = &(pictures[id]);
            }
            ++id;
        }

        // normalize the value pictures
        for (auto& p : pictures) {
            std::for_each(p.valueImage.begin(), p.valueImage.end(),
                [globalmin, globalmax](auto& n) { n = (n - globalmin) / (globalmax - globalmin); });
        }

        if (first) {
            this->cluster->insert(this->cluster->end(), this->leaves->begin(), this->leaves->end());
        }

        // add features to the feature vectors
        for (CLUSTERNODE* node : *this->cluster) {
            if (this->momentsmethode == 1)
                HierarchicalClustering::calculateImageMomentsValue(node);
            if (this->momentsmethode == 2)
                HierarchicalClustering::calculateColorMomentsValue(node);
            if (this->momentsmethode == 3) {
                switch (call.first) {
                case 0: {
                    if (features1.count(node->pic->pdbid) > 0) {
                        const std::vector<float>& feat = features1.at(node->pic->pdbid);
                        node->features->insert(node->features->end(), feat.begin(), feat.end());
                    } else {
                        core::utility::log::Log::DefaultLog.WriteWarn(
                            "Could not find feature vector in the feature list");
                    }
                    break;
                }
                case 1: {
                    if (features2.count(node->pic->pdbid) > 0) {
                        const std::vector<float>& feat = features2.at(node->pic->pdbid);
                        node->features->insert(node->features->end(), feat.begin(), feat.end());
                    } else {
                        core::utility::log::Log::DefaultLog.WriteWarn(
                            "Could not find feature vector in the feature list");
                    }
                    break;
                }
                case 2: {
                    if (features3.count(node->pic->pdbid) > 0) {
                        const std::vector<float>& feat = features3.at(node->pic->pdbid);
                        node->features->insert(node->features->end(), feat.begin(), feat.end());
                    } else {
                        core::utility::log::Log::DefaultLog.WriteWarn(
                            "Could not find feature vector in the feature list");
                    }
                    break;
                }
                default: {
                    if (features1.count(node->pic->pdbid) > 0) {
                        const std::vector<float>& feat = features1.at(node->pic->pdbid);
                        node->features->insert(node->features->end(), feat.begin(), feat.end());
                    } else {
                        core::utility::log::Log::DefaultLog.WriteWarn(
                            "Could not find feature vector in the feature list");
                    }
                    break;
                }
                }
            }
        }

        first = false;
    }

    this->clusterthedata();
}

HierarchicalClustering::HierarchicalClustering(HierarchicalClustering::CLUSTERNODE* node,
    const std::map<std::string, std::map<std::string, double>>& distmatrix, SIZE_T picturecount, int method, int mode,
    int linkage, int moments) {

    // Initalize Variables
    this->cluster = new std::vector<CLUSTERNODE*>();
    this->root = new std::vector<CLUSTERNODE*>();
    this->leaves = new std::vector<CLUSTERNODE*>();
    this->distancemethod = method;
    this->similaritymethod = method;
    this->mode = mode;
    this->linkagemethod = linkage;
    this->momentsmethode = moments;
    this->distanceMatrix = distmatrix;

    this->clusteringfinished = false;

    // Create Datastructure
    for (int index = 0; index < picturecount; index++) {
        CLUSTERNODE* tmpnode = &(node[index]);
        tmpnode->similiaritychildren = 1.0;

        // Init dtsiance Matrix
        tmpnode->distances = new std::vector<std::tuple<CLUSTERNODE*, double>>();

        this->leaves->push_back(tmpnode);
    }
    this->cluster->insert(this->cluster->end(), this->leaves->begin(), this->leaves->end());

    // Cluster the Data
    this->clusterthedata();
}

void HierarchicalClustering::calculateImageMomentsValue(CLUSTERNODE* node) {
    core::utility::log::Log::DefaultLog.WriteMsg(
        core::utility::log::Log::LEVEL_INFO, "\tAnalyzing %s", node->pic->path.c_str());

    PictureData* pic = node->pic;
    const auto& img = node->pic->valueImage;

    double m00 = 0.0;
    double m01 = 0.0;
    double m10 = 0.0;

    for (int y = 0; y < pic->height; y++) {
        for (int x = 0; x < pic->width; x++) {
            m00 += static_cast<double>(img[(y * pic->width) + x]);
            m10 += static_cast<double>(x) * static_cast<double>(img[(y * pic->width) + x]);
            m01 += static_cast<double>(y) * static_cast<double>(img[(y * pic->width) + x]);
        }
    }

    double xc = m10 / m00;
    double yc = m01 / m00;

    double müij[IMAGEORDER + 1][IMAGEORDER + 1] = {0.0};
    for (int i = 0; i <= IMAGEORDER; i++) {
        for (int j = 0; j <= IMAGEORDER; j++) {
            for (int y = 0; y < pic->height; y++) {
                for (int x = 0; x < pic->width; x++) {
                    if ((i + j <= 3) && !((i == 1 && j == 0) || (i == 0 && j == 1))) {
                        müij[i][j] += pow(static_cast<double>(x) - xc, i) * pow(static_cast<double>(y) - yc, j) *
                                      static_cast<double>(img[(y * pic->width) + x]);
                    }
                }
            }
        }
    }

    std::vector<double> nu;
    for (int i = 0; i <= IMAGEORDER; i++) {
        for (int j = 0; j <= IMAGEORDER; j++) {
            nu.push_back(müij[i][j] / (pow(müij[0][0], (1.0 + (static_cast<double>(i + j) / 2.0)))));
        }
    }

    double i1 = nu[2 * IMAGEORDER + 0] + nu[2];
    double i2 = pow(nu[2 * IMAGEORDER + 0] - nu[2], 2) + 4.0 * pow(nu[1 * IMAGEORDER + 1], 2);
    // double i3 =
    //    pow(nu[3 * IMAGEORDER + 0] - 3.0 * nu[1 * IMAGEORDER + 2], 2) + pow(3.0 * nu[2 * IMAGEORDER + 1] - nu[3],
    //    2);
    double i4 = pow(nu[3 * IMAGEORDER + 0] + nu[1 * IMAGEORDER + 2], 2) + pow(nu[2 * IMAGEORDER + 1] + nu[3], 2);
    double i5 =
        (nu[3 * IMAGEORDER + 0] - 3.0 * nu[1 * IMAGEORDER + 2]) * (nu[3 * IMAGEORDER + 0] + nu[1 * IMAGEORDER + 2]) *
            (pow((nu[3 * IMAGEORDER + 0] + nu[1 * IMAGEORDER + 2]), 2) - 3.0 * pow(nu[2 * IMAGEORDER + 1] + nu[3], 2)) +
        (3 * nu[2 * IMAGEORDER + 1] - nu[3]) * (nu[2 * IMAGEORDER + 1] + nu[3]) *
            (3.0 * pow(nu[3 * IMAGEORDER + 0] + nu[1 * IMAGEORDER + 2], 2) - pow(nu[2 * IMAGEORDER + 1] + nu[3], 2));
    double i6 = (nu[2 * IMAGEORDER + 0] - nu[2]) *
                    (pow(nu[3 * IMAGEORDER + 0] + nu[1 * IMAGEORDER + 2], 2) - pow(nu[2 * IMAGEORDER + 1] + nu[3], 2)) +
                4.0 * nu[1 * IMAGEORDER + 1] * (nu[3 * IMAGEORDER + 0] + nu[1 * IMAGEORDER + 2]) *
                    (nu[2 * IMAGEORDER + 1] + nu[3]);
    double i7 =
        (3.0 * nu[2 * IMAGEORDER + 1] - nu[3]) * (nu[3 * IMAGEORDER + 0] + nu[1 * IMAGEORDER + 2]) *
            (pow((nu[3 * IMAGEORDER + 0] + nu[1 * IMAGEORDER + 2]), 2) - 3.0 * pow(nu[2 * IMAGEORDER + 1] + nu[3], 2)) -
        (nu[3 * IMAGEORDER + 0] - 3 * nu[1 * IMAGEORDER + 2]) * (nu[2 * IMAGEORDER + 1] + nu[3]) *
            (3.0 * pow(nu[3 * IMAGEORDER + 0] + nu[1 * IMAGEORDER + 2], 2) - pow(nu[2 * IMAGEORDER + 1] + nu[3], 2));
    double i8 = nu[1 * IMAGEORDER + 1] *
                    (pow(nu[3 * IMAGEORDER + 0] + nu[1 * IMAGEORDER + 2], 2) - pow(nu[3] + nu[2 * IMAGEORDER + 1], 2)) -
                (nu[2 * IMAGEORDER + 0] - nu[2]) * (nu[3 * IMAGEORDER + 0] + nu[1 * IMAGEORDER + 2]) *
                    (nu[3] + nu[2 * IMAGEORDER + 1]);

    node->features->insert(node->features->end(), {i1, 0.0, 0.0, i2, i4, i5, i6, i7, i8});
    bla++;
    core::utility::log::Log::DefaultLog.WriteInfo("Calculated %u", bla);
}

void HierarchicalClustering::calculateColorMomentsValue(CLUSTERNODE* node) {
    core::utility::log::Log::DefaultLog.WriteMsg(
        core::utility::log::Log::LEVEL_INFO, "\tAnalyzing %s", node->pic->path.c_str());

    PictureData* pic = node->pic;

    double mean[3] = {0};
    double deviation[3] = {0};
    double skewness[3] = {0};

    double factor = static_cast<double>(pic->width * pic->height);
    auto picptr = pic->image->PeekDataAs<BYTE>();

    // Calculate Mean
    double summean[3] = {0};
    for (int i = 0; i < pic->height; i++) {
        for (int j = 0; j < pic->width * 3; j += 3) {
            // summean[0] += pic->rows[i][j];
            summean[0] += static_cast<double>(picptr[pic->width * i + j]);
            // summean[1] += pic->rows[i][j + 1];
            summean[1] += static_cast<double>(picptr[pic->width * i + j + 1]);
            // summean[2] += pic->rows[i][j + 2];
            summean[2] += static_cast<double>(picptr[pic->width * i + j + 2]);
        }
    }

    mean[0] = summean[0] / factor;
    mean[1] = summean[1] / factor;
    mean[2] = summean[2] / factor;

    // Calculate Deviation and Skewness
    double deviationsum[3] = {0.0};
    double skewnesssum[3] = {0.0};
    for (int i = 0; i < pic->height; i++) {
        for (int j = 0; j < pic->width * 3; j += 3) {
            deviationsum[0] += pow(static_cast<double>(picptr[pic->width * i + j]) - mean[0], 2);
            deviationsum[1] += pow(static_cast<double>(picptr[pic->width * i + j + 1]) - mean[1], 2);
            deviationsum[2] += pow(static_cast<double>(picptr[pic->width * i + j + 2]) - mean[2], 2);

            skewnesssum[0] += pow(static_cast<double>(picptr[pic->width * i + j]) - mean[0], 3);
            skewnesssum[1] += pow(static_cast<double>(picptr[pic->width * i + j + 1]) - mean[1], 3);
            skewnesssum[2] += pow(static_cast<double>(picptr[pic->width * i + j + 2]) - mean[2], 3);
        }
    }

    // Calculate Deviation
    deviation[0] = sqrt(deviationsum[0] / factor);
    deviation[1] = sqrt(deviationsum[1] / factor);
    deviation[2] = sqrt(deviationsum[2] / factor);

    skewness[0] = cbrt((skewnesssum[0] / factor));
    skewness[1] = cbrt((skewnesssum[1] / factor));
    skewness[2] = cbrt((skewnesssum[2] / factor));

    node->features->insert(node->features->end(),
        {mean[0], mean[1], mean[2], deviation[0], deviation[1], deviation[2], skewness[0], skewness[1], skewness[2]});

    bla++;
    core::utility::log::Log::DefaultLog.WriteInfo("Calculated %u", bla);
}

std::vector<double>* HierarchicalClustering::gray_scale_image(PictureData* pic) {

    std::vector<double>* result = new std::vector<double>();
    auto picptr = pic->image->PeekDataAs<BYTE>();
    auto val = pic->image->BytesPerPixel();

    for (int y = 0; y < pic->height; y++) {
        for (int x = 0; x < pic->width * 3; x += 3) {
            int red = picptr[y * pic->width + x];
            int green = picptr[y * pic->width + x + 1];
            int blue = picptr[y * pic->width + x + 2];

            result->push_back((0.21 * red) + (0.72 * green) + (0.07 * blue));
        }
    }
    return result;
}

double HierarchicalClustering::distance(std::vector<double>* X, std::vector<double>* Y, int r) {
    if (X == nullptr || Y == nullptr)
        return 1.0;

    double distance = 0.0;
    switch (r) {
    case 1:
        // City-Block-Mannhaten
        if (X->size() == Y->size()) {
            double summe = 0.0;
            for (int i = 0; i < X->size(); i++) {
                summe += abs((*X)[i] - (*Y)[i]);
            }
            distance = summe;
        } else {
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_ERROR, "Can not calculate Distance");
        }
        break;
    case 2:
        // Euclidian Distance
        if (X->size() == Y->size()) {
            double summe = 0.0;
            for (int i = 0; i < X->size(); i++) {
                summe += pow((*X)[i] - (*Y)[i], 2);
            }
            distance = sqrt(summe);

        } else {
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_ERROR, "Can not calculate Distance");
        }
        break;
    case 3:
        // L3
        if (X->size() == Y->size()) {
            double summe = 0.0;
            for (int i = 0; i < X->size(); i++) {
                summe += pow((*X)[i] - (*Y)[i], 3);
            }
            distance = cbrt(summe);
        } else {
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_ERROR, "Can not calculate Distance");
        }
        break;
    case 4:
        // Gower
    default:
        distance = 1.0;
    }
    return distance;
}

double HierarchicalClustering::nodeDistance(
    HierarchicalClustering::CLUSTERNODE* node1, HierarchicalClustering::CLUSTERNODE* node2, int distanceMode) {
    double distance = 0.0;

    if (node1 != nullptr && node2 != nullptr) {

        // we only have to check one direction as both directions are set != null at the same times
        if (node1->left == nullptr && node2->left == nullptr) { // case 2
            if (this->mode == DISTANCEMODE) {
                distance = this->distance(node1->features, node2->features, distanceMode);
            }
            if (this->mode == SIMILARITYMODE) {
                distance = this->similarity(node1->features, node2->features, distanceMode);
            }
            if (this->distanceMatrix.size() > 0) {
                std::string key1 = node1->pic->pdbid;
                std::string key2 = node2->pic->pdbid;
                if (this->distanceMatrix.find(key1) != this->distanceMatrix.end()) {
                    if (this->distanceMatrix.at(key1).find(key2) != this->distanceMatrix.at(key1).end()) {
                        distance = this->distanceMatrix.at(key1).at(key2);
                    }
                }
            }
        }
        if (node1->left == nullptr && node2->left != nullptr) { // case 3
            double leftdist = this->nodeDistance(node1, node2->left, distanceMode);
            double rightdist = this->nodeDistance(node1, node2->right, distanceMode);
            distance = 0.5 * (leftdist + rightdist);
        }
        if (node1->left != nullptr && node2->left == nullptr) { // case 4
            double leftdist = this->nodeDistance(node2, node1->left, distanceMode);
            double rightdist = this->nodeDistance(node2, node1->right, distanceMode);
            distance = 0.5 * (leftdist + rightdist);
        }
        if (node1->left != nullptr && node2->left != nullptr) { // case 5
            double d1 = this->nodeDistance(node1->left, node2->left, distanceMode);
            double d2 = this->nodeDistance(node1->left, node2->right, distanceMode);
            double d3 = this->nodeDistance(node1->right, node2->left, distanceMode);
            double d4 = this->nodeDistance(node1->right, node2->right, distanceMode);
            distance = 0.25 * (d1 + d2 + d3 + d4);
        }
        if (node1 == node2) { // case 1
            // intentionally empty as distance is zero
            distance = 0.0;
        }
    }

    return distance;
}

HierarchicalClustering::CLUSTERNODE* HierarchicalClustering::mergeCluster(
    CLUSTERNODE* cluster1, CLUSTERNODE* cluster2) {

    // Delete Old CLuster
    this->root->erase(find(this->root->begin(), this->root->end(), cluster1));
    this->root->erase(find(this->root->begin(), this->root->end(), cluster2));

    // Delete old Cluster from other Clusters
    for (CLUSTERNODE* node : *this->root) {
        for (std::tuple<CLUSTERNODE*, double> nodedistance : *node->distances) {
            if (std::get<0>(nodedistance) == cluster1) {
                // Delete tuple
                node->distances->erase(find(node->distances->begin(), node->distances->end(), nodedistance));
            }
        }

        for (std::tuple<CLUSTERNODE*, double> nodedistance : *node->distances) {
            if (std::get<0>(nodedistance) == cluster2) {
                // Delete tuple
                node->distances->erase(find(node->distances->begin(), node->distances->end(), nodedistance));
            }
        }
    }

    CLUSTERNODE* newnode = new CLUSTERNODE();
    newnode->id = cluster->size() + 1;
    newnode->level = (*this->cluster)[this->cluster->size() - 1]->level + 1;
    newnode->left = cluster1;
    newnode->right = cluster2;
    newnode->similiaritychildren = this->nodeDistance(cluster1, cluster2, this->similaritymethod);
    newnode->parent = nullptr;
    newnode->features = new std::vector<double>(cluster1->features->size(), 0.0);
    newnode->distances = new std::vector<std::tuple<CLUSTERNODE*, double>>();

    // Get all features from left childs
    auto cluster1Ptr = getLeavesOfNode(cluster1);
    for (const auto v : *cluster1Ptr) {
        std::transform(newnode->features->begin(), newnode->features->end(), v->features->begin(),
            newnode->features->begin(), std::plus<double>());
    }

    // Get all features from right childs
    auto cluster2Ptr = getLeavesOfNode(cluster2);
    for (const auto v : *cluster2Ptr) {
        std::transform(newnode->features->begin(), newnode->features->end(), v->features->begin(),
            newnode->features->begin(), std::plus<double>());
    }

    // Recalculate Featrues of Cluster
    double clength = static_cast<double>(cluster1Ptr->size() + cluster2Ptr->size());
    for (auto& v : *newnode->features) {
        v /= clength;
    }

    // FLAK THIS!!!111!!!1
    /*for (int i = 0; i < cluster1->features->size(); i++) {

        newnode->features->push_back(
            (((*cluster1->features)[i] * cluster1count) + ((*cluster2->features)[i] * cluster2count)) /
            (cluster1count + cluster2count));
    }*/

    newnode->pic = findNearestPicture(newnode->features, cluster1Ptr, cluster2Ptr);

    std::vector<CLUSTERNODE*>* leavesnew = getLeavesOfNode(newnode);
    // Recalculate Distances to all other Clusters
    // Change Method
    for (CLUSTERNODE* node : *this->root) {
        switch (this->linkagemethod) {
        case 1:
            // Centroide Linkage
            if (this->mode == DISTANCEMODE) {
                newnode->distances->push_back(std::make_tuple(node, this->nodeDistance(newnode, node, distancemethod)));
            } else if (this->mode == SIMILARITYMODE) {
                newnode->distances->push_back(
                    std::make_tuple(node, this->nodeDistance(newnode, node, similaritymethod)));
            }
            break;
        case 2:
            // Single Linkage
            // Get all Leaves of node
            if (this->mode == DISTANCEMODE) {
                std::vector<CLUSTERNODE*>* leaves = getLeavesOfNode(node);
                double mindistance = DBL_MAX;
                // Find nearest Leaf ==> min distance
                for (CLUSTERNODE* leafnew : *leavesnew) {
                    for (CLUSTERNODE* leaf : *leaves) {
                        double distance = this->nodeDistance(leafnew, leaf, distancemethod);
                        if (distance < mindistance) {
                            mindistance = distance;
                        }
                    }
                }
                newnode->distances->push_back(std::make_tuple(node, mindistance));
            } else if (this->mode == SIMILARITYMODE) {
                std::vector<CLUSTERNODE*>* leaves = getLeavesOfNode(node);
                double maxdistance = DBL_MAX * -1;
                // Find nearest Leaf ==> min distance
                for (CLUSTERNODE* leafnew : *leavesnew) {
                    for (CLUSTERNODE* leaf : *leaves) {
                        double distance = this->nodeDistance(leafnew, leaf, similaritymethod);
                        if (distance > maxdistance) {
                            maxdistance = distance;
                        }
                    }
                }
                newnode->distances->push_back(std::make_tuple(node, maxdistance));
            }
            break;
        case 3:
            // Avarage Linkage
            if (this->mode == DISTANCEMODE) {
                std::vector<CLUSTERNODE*>* leaves = getLeavesOfNode(node);
                double sum = 0;
                for (CLUSTERNODE* leafnew : *leavesnew) {
                    for (CLUSTERNODE* leaf : *leaves) {
                        sum += this->nodeDistance(leafnew, leaf, distancemethod);
                    }
                }
                double distance = sum / (leaves->size() * leavesnew->size());
                newnode->distances->push_back(std::make_tuple(node, distance));
            } else if (this->mode == SIMILARITYMODE) {
                std::vector<CLUSTERNODE*>* leaves = getLeavesOfNode(node);
                double sum = 0;
                for (CLUSTERNODE* leafnew : *leavesnew) {
                    for (CLUSTERNODE* leaf : *leaves) {
                        sum += this->nodeDistance(leafnew, leaf, similaritymethod);
                    }
                }
                double distance = sum / (leaves->size() * leavesnew->size());
                newnode->distances->push_back(std::make_tuple(node, distance));
            }
            break;
        }
    }

    float dist = 0.0f;
    if (this->mode == DISTANCEMODE) {
        dist = this->nodeDistance(cluster1, cluster2, distancemethod);
    } else if (this->mode == SIMILARITYMODE) {
        dist = this->nodeDistance(cluster1, cluster2, similaritymethod);
    }
    newnode->height = dist / 2.0f;

    // Update Parent of cluster1 and cluster2
    cluster1->parent = newnode;
    cluster2->parent = newnode;

    // Add to clusters and roots
    this->cluster->push_back(newnode);
    this->root->push_back(newnode);

    return newnode;
}

PictureData* HierarchicalClustering::findNearestPicture(std::vector<double>* features,
    std::vector<HierarchicalClustering::CLUSTERNODE*>* leftleaves,
    std::vector<HierarchicalClustering::CLUSTERNODE*>* rightleaves) {
    PictureData* pic = nullptr;
    double mindistance = DBL_MAX;
    for (CLUSTERNODE* node : *leftleaves) {
        if (distance(features, node->features) < mindistance) {
            mindistance = distance(features, node->features);
            pic = node->pic;
        }
    }
    for (CLUSTERNODE* node : *rightleaves) {
        if (distance(features, node->features) < mindistance) {
            mindistance = distance(features, node->features);
            pic = node->pic;
        }
    }
    return pic;
}

void HierarchicalClustering::dump_dot() {
    dump_dot("D://default.dot");
}

std::string HierarchicalClustering::getFeaturesString(std::vector<double>* features) {
    std::string featurestring = "";
    for (double feature : *features) {
        featurestring.append(std::to_string(feature));
        featurestring.append(";");
    }
    return featurestring.substr(0, featurestring.length() - 1);
}

void HierarchicalClustering::dump_dot(const std::filesystem::path& filename) {
    std::ofstream file;
    file.open(filename);

    core::utility::log::Log::DefaultLog.WriteMsg(
        core::utility::log::Log::LEVEL_INFO, "Creating .dot File at %s", filename);

    file << "graph g {" << std::endl;

    for (CLUSTERNODE* node : *this->cluster) {
        if (node->left == nullptr && node->right == nullptr) {

            file << node->id << "[label=\"" << std::filesystem::path(node->pic->path).filename() << "\", image=\""
                 << node->pic->path << "\", features=\"" << getFeaturesString(node->features) << "\", pca=\""
                 << getFeaturesString(node->pca2d)
                 << "\", shape=\"box\", scaleimage=true, fixedsize=true, width=6, height=3, level=" << node->level
                 << "]";
        } else {
            file << node->id << " -- {" << node->left->id << " " << node->right->id << "} "
                 << "[label=\"" << node->similiaritychildren << "\", image=\"" << node->pic->path << "\", features=\""
                 << getFeaturesString(node->features) << "\", shape=\"circle\", level=" << node->level << "]";
        }
        file << ";" << std::endl;
    }

    // Ranking of the nodes
    file << "{rank=same; ";
    int oldlevel = 0;
    for (CLUSTERNODE* node : *this->cluster) {
        // Level wechsel
        if (node->level > oldlevel) {
            oldlevel = node->level;
            file << "}" << std::endl << "{rank=same;" << node->id;
            // Erster Knoten
        } else if (node == (*this->cluster)[0]) {
            file << node->id;
            // Sonstige Knoten
        } else {
            file << ", " << node->id;
        }

        // Abschluss
        if (node == (*this->cluster)[this->cluster->size() - 1]) {
            file << "}";
        }
    }
    file << "}";
    file.close();

    // make svg
    std::string command = "dot -Tsvg " + filename.string() + " -o " + filename.string() + ".svg";
    system(command.c_str());
}

void HierarchicalClustering::dump_pca(const vislib::TString& filename) {
    std::ofstream file;
    std::string path(filename);
    file.open(path);

    core::utility::log::Log::DefaultLog.WriteMsg(
        core::utility::log::Log::LEVEL_INFO, "Creating PCA-CSV-File at %s", path);

    for (CLUSTERNODE* node : *this->leaves) {
        for (double comp : *node->pca2d) {
            file << comp << ";";
        }
        file << std::endl;
    }

    file.close();
}

bool HierarchicalClustering::finished() {
    return this->clusteringfinished;
}

void HierarchicalClustering::reanalyse(void) {

    this->clusteringfinished = false;

    // Delete old Clustering
    this->cluster->clear();
    this->cluster->insert(this->cluster->end(), this->leaves->begin(), this->leaves->end());

    // Getting Features of Pictures
    for (CLUSTERNODE* node : *this->cluster) {
        if (this->momentsmethode == 1)
            HierarchicalClustering::calculateImageMomentsValue(node);
        if (this->momentsmethode == 2)
            HierarchicalClustering::calculateColorMomentsValue(node);
    }

    // Cluster the Data
    this->clusterthedata();
}

void HierarchicalClustering::clusterthedata() {

    this->clusteringfinished = false;

    core::utility::log::Log::DefaultLog.WriteMsg(core::utility::log::Log::LEVEL_INFO,
        "Set up the Clustering with: Mode: %i, Distance: %i, Similarity: %i, Linkage: %i, Moments: %i", this->mode,
        this->distancemethod, this->similaritymethod, this->linkagemethod, this->momentsmethode);

    // Delete old Clustering
    this->cluster->clear();
    this->cluster->insert(this->cluster->end(), this->leaves->begin(), this->leaves->end());

    // Set up Root Nodes
    this->root->clear();
    this->root->insert(this->root->end(), this->leaves->begin(), this->leaves->end());

    // Reset Distance Matirx
    for (CLUSTERNODE* node : *this->leaves) {
        node->distances->clear();
    }

    // Set up Distance/Similarity Matrix
    if (this->mode == DISTANCEMODE) {
        for (CLUSTERNODE* node1 : *this->cluster) {
            for (CLUSTERNODE* node2 : *this->cluster) {
                node1->distances->push_back(
                    std::make_tuple(node2, this->nodeDistance(node1, node2, this->distancemethod)));
            }
        }

    } else if (this->mode == SIMILARITYMODE) {
        for (CLUSTERNODE* node1 : *this->cluster) {
            for (CLUSTERNODE* node2 : *this->cluster) {
                node1->distances->push_back(
                    std::make_tuple(node2, this->nodeDistance(node1, node2, this->similaritymethod)));
            }
        }
    }


    // Clustering
    core::utility::log::Log::DefaultLog.WriteMsg(core::utility::log::Log::LEVEL_INFO, "Clustering the Data");
    while (this->root->size() > 1) {
        if (this->mode == DISTANCEMODE) {
            // Search for minimum Distance Cluster
            CLUSTERNODE* minnode1 = nullptr;
            CLUSTERNODE* minnode2 = nullptr;
            double mindistance = DBL_MAX;

            for (CLUSTERNODE* node1 : *this->root) {
                for (std::tuple<CLUSTERNODE*, double> nodedistance : *node1->distances) {
                    if (node1 != std::get<0>(nodedistance)) {
                        if (std::get<1>(nodedistance) < mindistance) {
                            mindistance = std::get<1>(nodedistance);
                            minnode1 = node1;
                            minnode2 = std::get<0>(nodedistance);
                        }
                    }
                }
            }

            // Merge Cluster and Update Distance Matrix
            mergeCluster(minnode1, minnode2);

        } else if (this->mode == SIMILARITYMODE) {
            // Search for maximum similarity Cluster
            CLUSTERNODE* maxnode1 = nullptr;
            CLUSTERNODE* maxnode2 = nullptr;
            double maxdistance = DBL_MAX * -1;

            for (CLUSTERNODE* node1 : *this->root) {
                for (std::tuple<CLUSTERNODE*, double> nodedistance : *node1->distances) {
                    if (node1 != std::get<0>(nodedistance)) {
                        if (abs(std::get<1>(nodedistance)) > maxdistance) {
                            maxdistance = abs(std::get<1>(nodedistance));
                            maxnode1 = node1;
                            maxnode2 = std::get<0>(nodedistance);
                        }
                    }
                }
            }
            // Merge Cluster and Update Matrix
            mergeCluster(maxnode1, maxnode2);
        }
    }

    // Calculate PCA
    this->calculatePCA(this->cluster);
    this->dump_pca("D://pca.csv");

    this->clusteringfinished = true;
}

void HierarchicalClustering::setDistanceMethod(unsigned int method) {
    if (method < 4 && method > 0) {
        this->distancemethod = method;
    } else {
        this->distancemethod = 2; // standard methode
    }
}

void HierarchicalClustering::setSimilarityMethod(unsigned int method) {
    if (method < 4 && method > 0) {
        this->similaritymethod = method;
    } else {
        this->similaritymethod = 2; // standard methode
    }
}

void HierarchicalClustering::changeModeTo(short int mode) {
    this->mode = mode;
}

void HierarchicalClustering::setMoments(short int moments) {
    this->momentsmethode = moments;
}

std::vector<HierarchicalClustering::CLUSTERNODE*>* HierarchicalClustering::getLeaves() {
    return this->leaves;
}

double HierarchicalClustering::similarity(std::vector<double>* X, std::vector<double>* Y, int method) {

    if (X == nullptr || Y == nullptr)
        return 0.0f;

    double similar = 0.0;
    switch (method) {
    case 1:
        // Cosinus-Koeffizient
        if (X->size() == Y->size()) {
            double sumXY = 0.0;
            double sumX = 0.0;
            double sumY = 0.0;
            // Calculate summs
            for (int i = 0; i < X->size(); i++) {
                sumXY += (*X)[i] * (*Y)[i];
                sumX += pow((*X)[i], 2);
                sumY += pow((*Y)[i], 2);
            }
            // Calculate similarity
            similar = sumXY / (sqrt(sumX) * sqrt(sumY));
        } else {
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_ERROR, "Can not calculate Similarity");
        }
        break;
    case 2:
        // Dice-Koeffizient
        if (X->size() == Y->size()) {
            double sumXY = 0.0;
            double sumX = 0.0;
            double sumY = 0.0;
            // Calculate summs
            for (int i = 0; i < X->size(); i++) {
                sumXY += (*X)[i] * (*Y)[i];
                sumX += (*X)[i];
                sumY += (*Y)[i];
            }
            // Calculate similarity
            similar = 2.0 * sumXY / (sumX + sumY);

        } else {
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_ERROR, "Can not calculate Similarity");
        }
        break;
    case 3:
        // Jaccard-Koeffizient
        if (X->size() == Y->size()) {
            double sumXY = 0.0;
            double sumX = 0.0;
            double sumY = 0.0;
            // Calculate summs
            for (int i = 0; i < X->size(); i++) {
                sumXY += (*X)[i] * (*Y)[i];
                sumX += (*X)[i];
                sumY += (*Y)[i];
            }
            // Calculate similarity
            similar = sumXY / (sumX + sumY - sumXY);
        } else {
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_ERROR, "Can not calculate Similarity");
        }
        break;
    case 4:
        // Overlap-Koeffizient
        if (X->size() == Y->size()) {

        } else {
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_ERROR, "Can not calculate Similarity");
        }
        break;
    default:
        similar = 0.0;
    }
    return std::abs(similar);
}

HierarchicalClustering::CLUSTERNODE* HierarchicalClustering::getRoot() {
    return (*this->root)[0];
}

std::vector<HierarchicalClustering::CLUSTERNODE*>* HierarchicalClustering::getLeavesOfNode(
    HierarchicalClustering::CLUSTERNODE* node) {
    std::vector<CLUSTERNODE*>* tmp;
    std::vector<CLUSTERNODE*>* tmp2;

    if (node->left != nullptr && node->right != nullptr) {
        tmp = this->getLeavesOfNode(node->left);
        tmp2 = this->getLeavesOfNode(node->right);
        // Fusionate Vectors
        tmp->insert(tmp->end(), tmp2->begin(), tmp2->end());

    } else {
        // Leave found
        tmp = new std::vector<CLUSTERNODE*>(1, node);
    }
    return tmp;
}

HierarchicalClustering::CLUSTERNODE* HierarchicalClustering::getRootOfNode(HierarchicalClustering::CLUSTERNODE* node) {
    CLUSTERNODE* tmp = node;
    while (tmp->parent != nullptr) {
        tmp = tmp->parent;
    }
    return tmp;
}

void HierarchicalClustering::setLinkageMethod(unsigned int method) {
    this->linkagemethod = method;
}

void HierarchicalClustering::calculatePCA(std::vector<HierarchicalClustering::CLUSTERNODE*>* X) {

    // Load Data in Matrix
    int rowsCount = X->size();
    int columnCount = (*X)[0]->features->size();
    Eigen::MatrixXd data = Eigen::MatrixXd(rowsCount, columnCount);

    if (this->momentsmethode == 3) { // for ai features, just write something and exit
        for (int row = 0; row < rowsCount; row++) {
            CLUSTERNODE* node = (*X)[row];
            std::vector<double>* pca = new std::vector<double>();
            for (int column = 0; column < columnCount; column++) {
                pca->push_back(0.5);
            }
            node->pca2d = pca;
        }
        return;
    }

    for (int row = 0; row < rowsCount; row++) {
        std::vector<double>* rowVector = (*X)[row]->features;
        for (int column = 0; column < rowVector->size(); column++) {
            data(row, column) = (*rowVector)[column];
        }
    }

    // calculate mean for each column
    Eigen::VectorXd mean_vector(columnCount);
    mean_vector = data.colwise().mean();

    // prepare data
    // substract mean columnwise
    for (int col = 0; col < columnCount; col++) {
        data.col(col) -= Eigen::VectorXd::Constant(rowsCount, mean_vector(col));
    }

    // calculate CovarianceMatrix
    Eigen::MatrixXd covarianceMatrix = data;

    covarianceMatrix = covarianceMatrix.transpose() * covarianceMatrix;
    covarianceMatrix = covarianceMatrix / (float)(rowsCount - 1);

    // calculate Eigenvalues and Eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigSolver(covarianceMatrix);

    Eigen::VectorXd eigVal = eigSolver.eigenvalues().real();
    Eigen::MatrixXd eigVec = eigSolver.eigenvectors().real();

    // sort eigenvalues (with index): descending
    // each eigenvalue represents the variance
    typedef std::pair<float, int> eigenPair;
    std::vector<eigenPair> sorted;
    for (unsigned int i = 0; i < columnCount; ++i) {
        sorted.push_back(std::make_pair(static_cast<float>(eigVal(i)), i));
    }
    std::sort(sorted.begin(), sorted.end(), [&sorted](eigenPair& a, eigenPair& b) { return a.first > b.first; });

    // create Matrix out of sorted (and selected) eigenvectors
    Eigen::MatrixXd eigVecBasis = Eigen::MatrixXd(columnCount, 2);
    for (unsigned int i = 0; i < PCA; ++i) {
        eigVecBasis.col(i) = eigVec.col(sorted[i].second);
    }

    // calculate PCA
    Eigen::MatrixXd result = data * eigVecBasis;

    // write back PCA to CLUSTERNODES
    for (int row = 0; row < result.rows(); row++) {
        CLUSTERNODE* node = (*X)[row];
        std::vector<double>* pca = new std::vector<double>();
        for (int column = 0; column < result.cols(); column++) {
            pca->push_back(result(row, column));
        }
        node->pca2d = pca;
    }
}

std::vector<HierarchicalClustering::CLUSTERNODE*>* HierarchicalClustering::getClusterNodesOfNode(
    HierarchicalClustering::CLUSTERNODE* node) {

    std::vector<CLUSTERNODE*>* cluster = new std::vector<CLUSTERNODE*>();

    if (node->left == nullptr && node->right == nullptr) {
        return cluster;
    }

    std::vector<CLUSTERNODE*>* queue = new std::vector<CLUSTERNODE*>();
    queue->push_back(node);

    float rootheight = node->height;

    // Calculate cluster
    while (queue->size() > 0) {
        CLUSTERNODE* tmp = queue->back();
        queue->pop_back();
        // Check Distance of children
        if (tmp->left != nullptr && tmp->right != nullptr) {
            // Get max ditsance of leaves
            double dist = tmp->height;

            if (dist > rootheight * this->distancemultiplier) {
                queue->push_back(tmp->left);
                queue->push_back(tmp->right);
            } else {
                if (tmp == node) {
                    // Beider Cluster des azfrufenden clusters sind zu nahe beienander
                    cluster->push_back(tmp->left);
                    cluster->push_back(tmp->right);
                    tmp->left->clusterparent = node;
                    tmp->right->clusterparent = node;
                } else {
                    cluster->push_back(tmp);
                    tmp->clusterparent = node;
                }
            }

        } else {
            cluster->push_back(tmp);
            tmp->clusterparent = node;
        }
    }
    return cluster;
}

HierarchicalClustering::CLUSTERNODE* HierarchicalClustering::getClusterRootOfNode(
    HierarchicalClustering::CLUSTERNODE* node) {
    return node->clusterparent;
}

bool HierarchicalClustering::parentIs(
    HierarchicalClustering::CLUSTERNODE* ausgangsnode, HierarchicalClustering::CLUSTERNODE* searchfor) {
    do {
        if (ausgangsnode == searchfor) {
            return true;
        }
        ausgangsnode = ausgangsnode->parent;
    } while (ausgangsnode != nullptr);
    return false;
}

double HierarchicalClustering::getMaxDistanceOfLeavesToRoot() {
    double max = DBL_MAX * -1;
    for (CLUSTERNODE* node1 : *this->leaves) {
        double dist = this->nodeDistance(node1, this->getRoot());
        if (max < dist)
            max = dist;
    }
    return max;
}

void HierarchicalClustering::setDistanceMultiplier(float multiplier) {
    this->distancemultiplier = multiplier;
}

std::pair<float, float> HierarchicalClustering::loadValueImage(
    const std::filesystem::path& originalPicture, std::vector<float>& outValueImage) {
    auto newpath = originalPicture;
    newpath = newpath.parent_path();
    newpath.append(originalPicture.stem().string() + "_values.dat");
    newpath = newpath.make_preferred();

    auto filesize = std::filesystem::file_size(newpath);
    std::ifstream file(newpath, std::ios::binary);
    outValueImage.clear();
    outValueImage.resize(filesize / sizeof(float));
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(&outValueImage[0]), filesize);
        file.close();
        auto minmax = std::minmax_element(outValueImage.begin(), outValueImage.end());
        auto minele = std::abs(*minmax.first);
        for (auto& v : outValueImage) {
            v += minele;
            v = std::abs(v); // paranoia
        }
        auto newminmax = std::minmax_element(outValueImage.begin(), outValueImage.end());
        return std::make_pair(*newminmax.first, *newminmax.second);
    } else {
        core::utility::log::Log::DefaultLog.WriteError(
            "The file \"%s\" could not be opened for reading", newpath.c_str());
    }
}
