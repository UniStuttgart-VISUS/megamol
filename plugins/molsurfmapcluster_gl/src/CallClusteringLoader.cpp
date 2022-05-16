/*
 * CallSpheres.cpp
 *
 * Copyright (C) 2016 by Karsten Schatz
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "CallClusteringLoader.h"

using namespace megamol;
using namespace megamol::molsurfmapcluster;

/*
 * CallPNGPics::CallForGetData
 */
const unsigned int CallClusteringLoader::CallForGetData = 0;

/*
 * CallPNGPics::CallForGetExtent
 */
const unsigned int CallClusteringLoader::CallForGetExtent = 1;

/*
 * CallPNGPics::CallPNGPics
 */
CallClusteringLoader::CallClusteringLoader(void) : numberofleaves(0), nodes(nullptr) {}

/*
 * CallPNGPics::~CallPNGPics
 */
CallClusteringLoader::~CallClusteringLoader(void) {
    numberofleaves = 0;
    nodes = nullptr;
}

/*
 * CallPNGPics::Count
 */
SIZE_T CallClusteringLoader::Count(void) const { return this->numberofleaves; }

/*
 * CallPNGPics::getPNGPictures
 */
HierarchicalClustering::CLUSTERNODE* CallClusteringLoader::getLeaves(void) const { return this->nodes; }

/*
 * CallPNGPics::SetData
 */
void CallClusteringLoader::SetData(SIZE_T countofleaves, HierarchicalClustering::CLUSTERNODE* leaves) {
    this->numberofleaves = countofleaves;
    this->nodes = leaves;
}

/*
 * CallPNGPics::operator=
 */
CallClusteringLoader& CallClusteringLoader::operator=(const CallClusteringLoader& rhs) {
    AbstractGetData3DCall::operator=(rhs);
    this->numberofleaves = rhs.numberofleaves;
    this->nodes = rhs.nodes;
    return *this;
}
