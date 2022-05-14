#include "Clustering_2.h"

#include "CallClustering_2.h"
#include "image_calls/Image2DCall.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::molsurfmapcluster;

Clustering_2::Clustering_2(void)
    : Module()
    , getImageSlot("getImage", "Slot to retrieve the images to cluster over")
    , getImageSlot2("getImage2", "Slot to retrieve additional images to cluster over")
    , getImageSlot3("getImage3", "Slot to retrieve additional images to cluster over")
    , getImageSlot4("getImage4", "Slot to retrieve additional images to cluster over")
    , sendClusterSlot("sendCluster", "Slot to send the resulting clustering")
    , useMultipleMapsParam("useMultipleMaps", "Enables that the clustering uses multiple maps instead of one")
    , clusteringMethodSelectionParam("mode::clusteringMethod", "Selection of the used clustering method")
    , distanceMeasureSelectionParam("mode::distanceMeasure", "Selection of the used distance measure")
    , dataHashOffset(0) {
    // Callee slots
    this->sendClusterSlot.SetCallback(CallClustering_2::ClassName(),
        CallClustering_2::FunctionName(CallClustering_2::CallForGetExtent), &Clustering_2::GetExtentCallback);
    this->sendClusterSlot.SetCallback(CallClustering_2::ClassName(),
        CallClustering_2::FunctionName(CallClustering_2::CallForGetData), &Clustering_2::GetDataCallback);
    this->MakeSlotAvailable(&this->sendClusterSlot);

    // Caller slots
    this->getImageSlot.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->getImageSlot);

    this->getImageSlot2.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->getImageSlot2);

    this->getImageSlot3.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->getImageSlot3);

    this->getImageSlot4.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->getImageSlot4);

    // Param slots
    this->useMultipleMapsParam.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->useMultipleMapsParam);

    auto* clusteringMethodEnum = new core::param::EnumParam(static_cast<int>(ClusteringMethod::MOBILENETV2));
    clusteringMethodEnum->SetTypePair(static_cast<int>(ClusteringMethod::IMAGE_MOMENTS), "Image Moments");
    clusteringMethodEnum->SetTypePair(static_cast<int>(ClusteringMethod::COLOR_MOMENTS), "Color Moments");
    clusteringMethodEnum->SetTypePair(static_cast<int>(ClusteringMethod::MOBILENETV2), "MobileNetV2");
    this->clusteringMethodSelectionParam.SetParameter(clusteringMethodEnum);
    this->MakeSlotAvailable(&this->clusteringMethodSelectionParam);

    auto* distanceMeasureEnum = new core::param::EnumParam(static_cast<int>(DistanceMeasure::EUCLIDEAN_DISTANCE));
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::EUCLIDEAN_DISTANCE), "Euclidean Distance");
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::L3_DISTANCE), "L3 Distance");
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::COSINUS_DISTANCE), "Cosinus Similarity");
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::DICE_DISTANCE), "Dice Similarity");
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::JACCARD_DISTANCE), "Jaccard Similarity");
    this->distanceMeasureSelectionParam.SetParameter(distanceMeasureEnum);
    this->MakeSlotAvailable(&this->distanceMeasureSelectionParam);

    // Variables
    this->nodes = std::make_shared<std::vector<ClusterNode_2>>();
    this->lastDataHash = {0, 0, 0, 0};
}

Clustering_2::~Clustering_2(void) { this->Release(); }

bool Clustering_2::create(void) {
    // TODO
    return true;
}

void Clustering_2::release(void) {
    // TODO
}

bool Clustering_2::GetDataCallback(Call& call) {
    CallClustering_2* cc = dynamic_cast<CallClustering_2*>(&call);
    if (cc == nullptr) return false;

    if (this->recalculateClustering) {
        this->recalculateClustering = false;
        if (!runComputation()) return false;
    }

    // TODO copy data

    return true;
}

bool Clustering_2::GetExtentCallback(Call& call) {
    CallClustering_2* cc = dynamic_cast<CallClustering_2*>(&call);
    if (cc == nullptr) return false;

    if (this->checkParameterDirtyness()) {
        this->recalculateClustering = true;
    }

    std::vector<std::pair<image_calls::Image2DCall*, int>> calls;
    calls.push_back(std::make_pair(this->getImageSlot.CallAs<image_calls::Image2DCall>(), 0));
    {
        image_calls::Image2DCall* imc2 = this->getImageSlot2.CallAs<image_calls::Image2DCall>();
        image_calls::Image2DCall* imc3 = this->getImageSlot3.CallAs<image_calls::Image2DCall>();
        image_calls::Image2DCall* imc4 = this->getImageSlot4.CallAs<image_calls::Image2DCall>();

        if (imc2 != nullptr) calls.push_back(std::make_pair(imc2, 1));
        if (imc3 != nullptr) calls.push_back(std::make_pair(imc3, 2));
        if (imc4 != nullptr) calls.push_back(std::make_pair(imc4, 3));
    }
    // if the first one is not connected, fail
    if (calls.at(0).first == nullptr) return false;

    // get the metadata for all relevant calls
    if (this->useMultipleMapsParam.Param<param::BoolParam>()->Value()) {
        for (const auto& c : calls) {
            if (!(*c.first)(image_calls::Image2DCall::CallForGetMetaData)) return false;
            if (this->lastDataHash.at(c.second) != c.first->DataHash()) {
                this->lastDataHash.at(c.second) = c.first->DataHash();
                this->recalculateClustering = true;
            }
        }
    } else {
        if (!(*calls.at(0).first)(image_calls::Image2DCall::CallForGetMetaData)) return false;
        if (this->lastDataHash.at(calls.at(0).second) != calls.at(0).first->DataHash()) {
            this->lastDataHash.at(calls.at(0).second) = calls.at(0).first->DataHash();
            this->recalculateClustering = true;
        }
    }

    if (this->recalculateClustering) {
        this->dataHashOffset++;
    }

    ClusteringMetaData meta;
    meta.dataHash = this->dataHashOffset;
    for (const auto& val : this->lastDataHash) {
        meta.dataHash += val;
    }
    meta.numLeafNodes = calls.at(0).first->GetImageCount();
    cc->SetMetaData(meta);

    return true;
}

bool Clustering_2::checkParameterDirtyness(void) {
    bool result = false;
    if (this->clusteringMethodSelectionParam.IsDirty()) {
        this->clusteringMethodSelectionParam.ResetDirty();
        result = true;
    }
    if (this->distanceMeasureSelectionParam.IsDirty()) {
        this->distanceMeasureSelectionParam.ResetDirty();
        result = true;
    }
    if (this->useMultipleMapsParam.IsDirty()) {
        this->useMultipleMapsParam.ResetDirty();
        result = true;
    }
    return result;
}

bool Clustering_2::runComputation(void) {
    //if (!this->calculateFeatureVectors) return false;
    //if (!this->clusterImages) return false;
    return true;
}


bool Clustering_2::calculateFeatureVectors(void) {
    // TODO implement
    return true;
}

bool Clustering_2::clusterImages(void) {
    // TODO implement
    return true;
}
