/*
 * CallCluster.cpp
 *
 * Copyright (C) 2019 by Tobias Baur
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "CallCluster.h"

using namespace megamol;
using namespace megamol::molsurfmapcluster;

/*
 * CallClustering::CallForGetData
 */
const unsigned int CallClustering::CallForGetData = 0;

/*
 * CallClustering::CallForGetExtent
 */
const unsigned int CallClustering::CallForGetExtent = 1;

/*
 * CallClustering::CallSpheres
 */
CallClustering::CallClustering(void) : clustering(nullptr) {}

/*
 * CallClustering::~CallSpheres
 */
CallClustering::~CallClustering(void) {
    if (this->clustering != nullptr)
        this->clustering->~HierarchicalClustering();
}

/*
 * CallClustering::setClustering
 */
void CallClustering::setClustering(HierarchicalClustering* clustering) {
    this->clustering = clustering;
}

/*
 * CallClustering::getClustering
 */
HierarchicalClustering* CallClustering::getClustering(void) {
    return this->clustering;
}

/*
 * CallClustering::operator=
 */
CallClustering& CallClustering::operator=(const CallClustering& rhs) {
    AbstractGetData3DCall::operator=(rhs);
    this->clustering = rhs.clustering;
    return *this;
}
