/*
 * CallClusterPosition.cpp
 *
 * Copyright (C) 2019 by Tobias Baur
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "CallClusterPosition.h"

using namespace megamol;
using namespace megamol::molsurfmapcluster;

/*
 * CallClustering::CallForGetData
 */
const unsigned int CallClusterPosition::CallForGetData = 0;

/*
 * CallClustering::CallForGetExtent
 */
const unsigned int CallClusterPosition::CallForGetExtent = 1;

/*
 * CallClustering::CallSpheres
 */
CallClusterPosition::CallClusterPosition(void) : position(nullptr), colors(nullptr) {}

/*
 * CallClustering::~CallSpheres
 */
CallClusterPosition::~CallClusterPosition(void) {
    this->position = nullptr;
    this->colors = nullptr;
}

/*
 * CallClustering::setClustering
 */
void CallClusterPosition::setPosition(HierarchicalClustering::CLUSTERNODE* position) {
    this->position = position;
}

/*
 * CallClustering::getClustering
 */
HierarchicalClustering::CLUSTERNODE* CallClusterPosition::getPosition(void) {
    return this->position;
}

/*
 * CallClustering::getClusterColors
 */
std::vector<std::tuple<HierarchicalClustering::CLUSTERNODE*, ClusterRenderer::RGBCOLOR*>*>*
CallClusterPosition::getClusterColors() {
    return this->colors;
}

/*
 * CallClustering::setClusterColors
 */
void CallClusterPosition::setClusterColors(
    std::vector<std::tuple<HierarchicalClustering::CLUSTERNODE*, ClusterRenderer::RGBCOLOR*>*>* colors) {
    this->colors = colors;
}


/*
 * CallClustering::operator=
 */
CallClusterPosition& CallClusterPosition::operator=(const CallClusterPosition& rhs) {
    AbstractGetData3DCall::operator=(rhs);
    this->position = rhs.position;
    this->colors = rhs.colors;
    return *this;
}
