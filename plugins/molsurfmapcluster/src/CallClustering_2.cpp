#include "CallClustering_2.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::molsurfmapcluster;

/*
 * CallClustering_2::CallForGetData
 */
const unsigned int CallClustering_2::CallForGetData = 0;

/*
 * CallClustering_2::CallForGetExtent
 */
const unsigned int CallClustering_2::CallForGetExtent = 1;

/*
 * CallClustering_2::CallClustering_2
 */
CallClustering_2::CallClustering_2(void) : Call() {
    // intentionally empty
}

/*
 * CallClustering_2::~CallClustering_2
 */
CallClustering_2::~CallClustering_2(void) {
    // intentionally empty
}

/*
 * CallClustering_2::GetData
 */
ClusteringData CallClustering_2::GetData(void) const { return this->data; }

/*
 * CallClustering_2::GetMetaData
 */
ClusteringMetaData CallClustering_2::GetMetaData(void) const { return this->metadata; }

/*
 * CallClustering_2::SetData
 */
void CallClustering_2::SetData(const ClusteringData& data) { this->data = data; }

/*
 * CallClustering_2::SetMetaData
 */
void CallClustering_2::SetMetaData(const ClusteringMetaData& metadata) { this->metadata = metadata; }
