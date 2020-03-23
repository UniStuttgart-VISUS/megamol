#include "Clustering_2.h"

#include "image_calls/Image2DCall.h"
#include "CallClustering_2.h"

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
    // TODO
    return true;
}
