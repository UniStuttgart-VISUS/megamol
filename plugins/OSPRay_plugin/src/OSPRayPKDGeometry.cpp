/*
* OSPRayPKDGeometry.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayPKDGeometry.h"
#include "vislib/forceinline.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/EnumParam.h"
#include "vislib/sys/Log.h"
#include "mmcore/Call.h"

#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallClipPlane.h"

using namespace megamol::ospray;


VISLIB_FORCEINLINE float floatFromVoidArray(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
    //const float* parts = static_cast<const float*>(p.GetVertexData());
    //return parts[index * stride + offset];
    return static_cast<const float*>(p.GetVertexData())[index];
}

VISLIB_FORCEINLINE unsigned char byteFromVoidArray(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
    return static_cast<const unsigned char*>(p.GetVertexData())[index];
}

typedef float(*floatFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);
typedef unsigned char(*byteFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);


OSPRayPKDGeometry::OSPRayPKDGeometry(void) :
    AbstractOSPRayStructure(),
    getDataSlot("getdata", "Connects to the data source"),

    particleList("ParticleList", "Switches between particle lists")
{

    this->getDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->particleList << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->particleList);
}


bool OSPRayPKDGeometry::readData(megamol::core::Call &call) {

    // fill material container
    this->processMaterial();

    // read Data, calculate  shape parameters, fill data vectors
    CallOSPRayStructure *os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::core::moldyn::MultiParticleDataCall *cd = this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();

    this->structureContainer.dataChanged = false;
    if (cd == NULL) return false;
	cd->SetTimeStamp(os->getTime());
    cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    if (this->datahash != cd->DataHash() || this->time != os->getTime() || this->InterfaceIsDirty()) {
        this->datahash = cd->DataHash();
        this->time = os->getTime();
        this->structureContainer.dataChanged = true;
    } else {
        return true;
    }

    if (this->particleList.Param<core::param::IntParam>()->Value() >(cd->GetParticleListCount() - 1)) {
        this->particleList.Param<core::param::IntParam>()->SetValue(0);
    }

    if (!(*cd)(1)) return false;
    if (!(*cd)(0)) return false;

    core::moldyn::MultiParticleDataCall::Particles &parts = cd->AccessParticles(this->particleList.Param<core::param::IntParam>()->Value());

    unsigned int partCount = parts.GetCount();
    float globalRadius = parts.GetGlobalRadius();

    size_t vertexLength = 3;
    size_t colorLength = 0;

    // Write stuff into the structureContainer
    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::PKD;
    this->structureContainer.raw = std::make_shared<const void*>(std::move(parts.GetVertexData()));
    this->structureContainer.vertexLength = vertexLength;
    this->structureContainer.colorLength = colorLength;
    this->structureContainer.partCount = partCount;
    this->structureContainer.globalRadius = globalRadius;

    return true;
}


OSPRayPKDGeometry::~OSPRayPKDGeometry() {
    this->Release();
}

bool OSPRayPKDGeometry::create() {
    return true;
}

void OSPRayPKDGeometry::release() {

}

/*
ospray::OSPRayPKDGeometry::InterfaceIsDirty()
*/
bool OSPRayPKDGeometry::InterfaceIsDirty() {
    if (
        this->particleList.IsDirty()
        ) {
        this->particleList.ResetDirty();
        return true;
    } else {
        return false;
    }
}


bool OSPRayPKDGeometry::getExtends(megamol::core::Call &call) {
    CallOSPRayStructure *os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::core::moldyn::MultiParticleDataCall *cd = this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();
    
    if (cd == NULL) return false;
    cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    // if (!(*cd)(1)) return false; // floattable returns flase at first attempt and breaks everything
    (*cd)(1);
    this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes>(cd->AccessBoundingBoxes());
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}