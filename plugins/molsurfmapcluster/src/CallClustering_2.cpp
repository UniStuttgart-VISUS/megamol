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
CallClustering_2::CallClustering_2(void) : Call(), datahash(0) {
    // intentionally empty
}

/*
 * CallClustering_2::~CallClustering_2
 */
CallClustering_2::~CallClustering_2(void) {
    // intentionally empty
}

/*
 * CallClustering_2::GetDataHash
 */
uint64_t CallClustering_2::GetDataHash(void) const { return this->datahash; }

/*
 * CallClustering_2::SetDataHash
 */
void CallClustering_2::SetDataHash(uint64_t datahash) { this->datahash = datahash; }
