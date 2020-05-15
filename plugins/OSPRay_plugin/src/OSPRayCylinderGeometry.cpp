/*
* OSPRayCylinderGeometry.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayCylinderGeometry.h"
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


typedef float(*floatFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);
typedef unsigned char(*byteFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);


OSPRayCylinderGeometry::OSPRayCylinderGeometry(void) :
    AbstractOSPRayStructure(),
    getDataSlot("getdata", "Connects to the data source"),

    particleList("ParticleList", "Switches between particle lists")
{

    this->getDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->particleList << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->particleList);
}


bool OSPRayCylinderGeometry::readData(megamol::core::Call &call) {

    // fill material container
    this->processMaterial();

    // get transformation parameter
    this->processTransformation();

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

    if (!(*cd)(1)) return false;
    if (!(*cd)(0)) return false;
    if (cd->GetParticleListCount() == 0) return false;

    if (this->particleList.Param<core::param::IntParam>()->Value() > (cd->GetParticleListCount() - 1)) {
        this->particleList.Param<core::param::IntParam>()->SetValue(0);
    }

    core::moldyn::MultiParticleDataCall::Particles &parts = cd->AccessParticles(this->particleList.Param<core::param::IntParam>()->Value());

    unsigned int partCount = parts.GetCount();
    float globalRadius = parts.GetGlobalRadius();

    size_t vertexLength;
    size_t colorLength;

    // Vertex data type check
    if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) {
        vertexLength = 3;
    } else if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
        vertexLength = 4;
    }

    // Color data type check
    if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA) {
        colorLength = 4;
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
        colorLength = 1;
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB) {
        colorLength = 3;
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA) {
        colorLength = 4;
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE) {
        colorLength = 0;
    }

    int vstride = parts.GetVertexDataStride();
    if (parts.GetVertexDataStride() == 0) {
        vstride = core::moldyn::MultiParticleDataCall::Particles::VertexDataSize[parts.GetVertexDataType()];
    }

    if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE &&
        parts.GetColourDataType() != core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE) {
        vislib::sys::Log::DefaultLog.WriteError("Only color data is not allowed.");
    }

    // Write stuff into the structureContainer
    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::CYLINDERS;
    this->structureContainer.raw = parts.GetVertexData();
    this->structureContainer.vertexLength = vertexLength;
    this->structureContainer.vertexStride = vstride;
    this->structureContainer.colorLength = colorLength;
    this->structureContainer.colorStride = parts.GetColourDataStride();
    this->structureContainer.partCount = partCount;
    this->structureContainer.globalRadius = globalRadius;
    this->structureContainer.mmpldColor = parts.GetColourDataType();

    return true;
}


OSPRayCylinderGeometry::~OSPRayCylinderGeometry() {
    this->Release();
}

bool OSPRayCylinderGeometry::create() {
    return true;
}

void OSPRayCylinderGeometry::release() {

}

/*
ospray::OSPRayCylinderGeometry::InterfaceIsDirty()
*/
bool OSPRayCylinderGeometry::InterfaceIsDirty() {
    if (
        this->particleList.IsDirty()
        ) {
        this->particleList.ResetDirty();
        return true;
    } else {
        return false;
    }
}


bool OSPRayCylinderGeometry::getExtends(megamol::core::Call &call) {
    CallOSPRayStructure *os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::core::moldyn::MultiParticleDataCall *cd = this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();
    
    if (cd == NULL) return false;
    cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    // if (!(*cd)(1)) return false; // table returns flase at first attempt and breaks everything
    (*cd)(1);
    this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes_2>();
    this->extendContainer.boundingBox->SetBoundingBox(cd->AccessBoundingBoxes().ObjectSpaceBBox());
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}