#include "Clustering_2.h"

#include "image_calls/Image2DCall.h"
#include "CallClustering_2.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::molsurfmapcluster;

Clustering_2::Clustering_2(void)
    : Module()
    , getImageSlot("getImage", "Slot to retrieve the images to cluster over")
    , sendClusterSlot("sendCluster", "Slot to send the resulting clustering")
    , dataHashOffset(0)
    , lastDataHash(0) {
    // Callee slots
    this->sendClusterSlot.SetCallback(CallClustering_2::ClassName(),
        CallClustering_2::FunctionName(CallClustering_2::CallForGetExtent), &Clustering_2::GetExtentCallback);
    this->sendClusterSlot.SetCallback(CallClustering_2::ClassName(),
        CallClustering_2::FunctionName(CallClustering_2::CallForGetData), &Clustering_2::GetDataCallback);
    this->MakeSlotAvailable(&this->sendClusterSlot);

    // Caller slot
    this->getImageSlot.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->getImageSlot);
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
