#include "geometry_calls/SimpleSphericalParticles.h"


namespace megamol::geocalls {


unsigned int SimpleSphericalParticles::VertexDataSize[] = {0, 12, 16, 6, 24};

unsigned int SimpleSphericalParticles::ColorDataSize[] = {0, 3, 4, 12, 16, 4, 8, 8};

unsigned int SimpleSphericalParticles::DirDataSize[] = {0, 12};

unsigned int SimpleSphericalParticles::IDDataSize[] = {0, 4, 8};


/*
 * SimpleSphericalParticles::SimpleSphericalParticles
 */
SimpleSphericalParticles::SimpleSphericalParticles()
        : colDataType(COLDATA_NONE)
        , colPtr(nullptr)
        , colStride(0)
        , dirDataType(DIRDATA_NONE)
        , dirPtr(nullptr)
        , dirStride(0)
        , count(0)
        , maxColI(1.0f)
        , minColI(0.0f)
        , radius(0.5f)
        , particleType(0)
        , vertDataType(VERTDATA_NONE)
        , vertPtr(nullptr)
        , vertStride(0)
        , disabledNullChecks(true)
        , isVAO(false)
        , clusterInfos(nullptr)
        , idDataType{IDDATA_NONE}
        , idPtr{nullptr}
        , idStride{0} {
    this->col[0] = 255;
    this->col[1] = 0;
    this->col[2] = 0;
    this->col[3] = 255;

    this->par_store_->SetVertexData(VERTDATA_NONE, nullptr);
    this->par_store_->SetColorData(COLDATA_NONE, nullptr);
    this->par_store_->SetDirData(DIRDATA_NONE, nullptr);
    this->par_store_->SetIDData(IDDATA_NONE, nullptr);
}


/*
 * SimpleSphericalParticles::SimpleSphericalParticles
 */
SimpleSphericalParticles::SimpleSphericalParticles(const SimpleSphericalParticles& src) {
    *this = src;
}


/*
 * SimpleSphericalParticles::~SimpleSphericalParticles
 */
SimpleSphericalParticles::~SimpleSphericalParticles() {
    this->colDataType = COLDATA_NONE;
    this->colPtr = nullptr; // DO NOT DELETE
    this->count = 0;
    this->vertDataType = VERTDATA_NONE;
    this->vertPtr = nullptr; // DO NOT DELETE
    this->dirDataType = DIRDATA_NONE;
    this->dirPtr = nullptr; // DO NOT DELETE
    this->idDataType = IDDATA_NONE;
    this->idPtr = nullptr;
}


/*
 * SimpleSphericalParticles::operator=
 */
SimpleSphericalParticles& SimpleSphericalParticles::operator=(const SimpleSphericalParticles& rhs) {
    this->col[0] = rhs.col[0];
    this->col[1] = rhs.col[1];
    this->col[2] = rhs.col[2];
    this->col[3] = rhs.col[3];
    this->colDataType = rhs.colDataType;
    this->colPtr = rhs.colPtr;
    this->colStride = rhs.colStride;
    this->count = rhs.count;
    this->maxColI = rhs.maxColI;
    this->minColI = rhs.minColI;
    this->radius = rhs.radius;
    this->particleType = rhs.particleType;
    this->vertDataType = rhs.vertDataType;
    this->vertPtr = rhs.vertPtr;
    this->vertStride = rhs.vertStride;
    this->disabledNullChecks = rhs.disabledNullChecks;
    this->clusterInfos = rhs.clusterInfos;
    this->dirDataType = rhs.dirDataType;
    this->dirPtr = rhs.dirPtr;
    this->dirStride = rhs.dirStride;
    this->idDataType = rhs.idDataType;
    this->idPtr = rhs.idPtr;
    this->idStride = rhs.idStride;
    this->par_store_ = rhs.par_store_;
    this->wsBBox = rhs.wsBBox;
    return *this;
}


/*
 * SimpleSphericalParticles::operator==
 */
bool SimpleSphericalParticles::operator==(const SimpleSphericalParticles& rhs) const {
    return ((this->col[0] == rhs.col[0]) && (this->col[1] == rhs.col[1]) && (this->col[2] == rhs.col[2]) &&
            (this->col[3] == rhs.col[3]) && (this->colDataType == rhs.colDataType) && (this->colPtr == rhs.colPtr) &&
            (this->colStride == rhs.colStride) && (this->count == rhs.count) && (this->maxColI == rhs.maxColI) &&
            (this->minColI == rhs.minColI) && (this->radius == rhs.radius) &&
            (this->vertDataType == rhs.vertDataType) && (this->vertPtr == rhs.vertPtr) &&
            (this->vertStride == rhs.vertStride) && (this->clusterInfos == rhs.clusterInfos) &&
            (this->dirDataType == rhs.dirDataType) && (this->dirPtr == rhs.dirPtr) &&
            (this->dirStride == rhs.dirStride) && (this->idDataType == rhs.idDataType) && (this->idPtr == rhs.idPtr) &&
            (this->idStride == rhs.idStride) && (this->wsBBox == rhs.wsBBox));
}

} // namespace megamol::geocalls
