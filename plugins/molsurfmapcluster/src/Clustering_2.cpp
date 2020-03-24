#include "Clustering_2.h"

#include "CallClustering_2.h"
#include "image_calls/Image2DCall.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/BoolParam.h"

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
    , linkageMethodSelectionParam("mode::linkageMethod", "Selection of the used linkage method")
    , dataHashOffset(0)
    , lastDataHash(0) {
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

    auto* clusteringMethodEnum = new core::param::EnumParam(0);
    clusteringMethodEnum->SetTypePair(static_cast<int>(ClusteringMethod::IMAGE_MOMENTS), "Image Moments");
    clusteringMethodEnum->SetTypePair(static_cast<int>(ClusteringMethod::COLOR_MOMENTS), "Color Moments");
    this->clusteringMethodSelectionParam.SetParameter(clusteringMethodEnum);
    this->MakeSlotAvailable(&this->clusteringMethodSelectionParam);

    auto* distanceMeasureEnum = new core::param::EnumParam(0);
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::EUCLIDEAN_DISTANCE), "Euclidean Distance");
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::L3_DISTANCE), "L3 Distance");
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::COSINUS_DISTANCE), "Cosinus Similarity");
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::DICE_DISTANCE), "Dice Similarity");
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::JACCARD_DISTANCE), "Jaccard Similarity");
    this->distanceMeasureSelectionParam.SetParameter(distanceMeasureEnum);
    this->MakeSlotAvailable(&this->distanceMeasureSelectionParam);

    auto* linkageMethodEnum = new core::param::EnumParam(0);
    linkageMethodEnum->SetTypePair(static_cast<int>(LinkageMethod::CENTROID_LINKAGE), "Centroid");
    linkageMethodEnum->SetTypePair(static_cast<int>(LinkageMethod::SINGLE_LINKAGE), "Single");
    linkageMethodEnum->SetTypePair(static_cast<int>(LinkageMethod::AVERAGE_LINKAGE), "Average");
    this->linkageMethodSelectionParam.SetParameter(linkageMethodEnum);
    this->MakeSlotAvailable(&this->linkageMethodSelectionParam);

    // Variables
    this->nodes = std::make_shared<std::vector<ClusterNode_2>>();
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
    // TODO
    return true;
}

bool Clustering_2::GetExtentCallback(Call& call) {
    std::vector<image_calls::Image2DCall*> calls;
    calls.push_back(this->getImageSlot.CallAs<image_calls::Image2DCall>());

    {
        image_calls::Image2DCall* imc2 = this->getImageSlot2.CallAs<image_calls::Image2DCall>();
        image_calls::Image2DCall* imc3 = this->getImageSlot3.CallAs<image_calls::Image2DCall>();
        image_calls::Image2DCall* imc4 = this->getImageSlot4.CallAs<image_calls::Image2DCall>();

        if (imc2 != nullptr) calls.push_back(imc2);
        if (imc3 != nullptr) calls.push_back(imc3);
        if (imc4 != nullptr) calls.push_back(imc4);
    }
    // if the first one is not connected, fail
    if (calls.at(0) == nullptr) return false;

    // get the metadata for all calls
    for (const auto& c : calls) {
        if (!(*c)(image_calls::Image2DCall::CallForGetMetaData)) return false;
    }

    return true;
}
