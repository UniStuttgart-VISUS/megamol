/*
* OSPRayNHSphereGeometry.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayNHSphereGeometry.h"
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


OSPRayNHSphereGeometry::OSPRayNHSphereGeometry(void) :
    AbstractOSPRayStructure(),
    getDataSlot("getdata", "Connects to the data source"),
    getTFSlot("gettransferfunction", "Connects to the transfer function module"),
    getClipPlaneSlot("getclipplane", "Connects to a clipping plane module"),

    particleList("ParticleList", "Switches between particle lists")
{

    this->getDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getClipPlaneSlot.SetCompatibleCall<core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);

    this->particleList << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->particleList);
}


bool OSPRayNHSphereGeometry::readData(megamol::core::Call &call) {

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

    size_t vertexLength;
    size_t colorLength;

    // Vertex data type check
    if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) {
        vertexLength = 3 * sizeof(float);
    } else if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
        vertexLength = 4 * sizeof(float);
    }

    // Color data type check
    if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA) {
        colorLength = 4 * sizeof(float);
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
        colorLength = 1 * sizeof(float);
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB) {
        colorLength = 3 * sizeof(unsigned char);
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA) {
        colorLength = 4 * sizeof(unsigned char);
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE) {
        colorLength = 0;
    }


    // clipPlane setup
    std::vector<float> clipDat(4);
    std::vector<float> clipCol(4);
    this->getClipData(clipDat.data(), clipCol.data());


    // Write stuff into the structureContainer
    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::NHSPHERES;
    this->structureContainer.raw = std::make_shared<const void*>(std::move(parts.GetVertexData()));
    this->structureContainer.vertexLength = vertexLength;
    this->structureContainer.colorLength = colorLength;
    this->structureContainer.partCount = partCount;
    this->structureContainer.globalRadius = globalRadius;
    this->structureContainer.clipPlaneData = std::make_shared<std::vector<float>>(std::move(clipDat));
    this->structureContainer.clipPlaneColor = std::make_shared<std::vector<float>>(std::move(clipCol));


    // material container
    CallOSPRayMaterial *cm = this->getMaterialSlot.CallAs<CallOSPRayMaterial>();
    if (cm != NULL) {
        auto gmp = cm->getMaterialParameter();
        if (gmp->isValid) {
            this->structureContainer.materialContainer = cm->getMaterialParameter();
        }
    } else {
        this->structureContainer.materialContainer = NULL;
    }



    return true;
}


OSPRayNHSphereGeometry::~OSPRayNHSphereGeometry() {
    this->Release();
}

bool OSPRayNHSphereGeometry::create() {
    return true;
}

void OSPRayNHSphereGeometry::release() {

}

/*
ospray::OSPRayNHSphereGeometry::InterfaceIsDirty()
*/
bool OSPRayNHSphereGeometry::InterfaceIsDirty() {
    CallOSPRayMaterial *cm = this->getMaterialSlot.CallAs<CallOSPRayMaterial>();
    cm->getMaterialParameter();
    if (
        cm->InterfaceIsDirty() ||
        this->particleList.IsDirty()
        ) {
        this->particleList.ResetDirty();
        return true;
    } else {
        return false;
    }
}

/*
* ospray::OSPRayNHSphereGeometry::getClipData
*/
void OSPRayNHSphereGeometry::getClipData(float *clipDat, float *clipCol) {
    megamol::core::view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<megamol::core::view::CallClipPlane>();
    if ((ccp != NULL) && (*ccp)()) {
        clipDat[0] = ccp->GetPlane().Normal().X();
        clipDat[1] = ccp->GetPlane().Normal().Y();
        clipDat[2] = ccp->GetPlane().Normal().Z();
        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
        clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
        clipCol[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
        clipCol[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
        clipCol[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;
        clipCol[3] = static_cast<float>(ccp->GetColour()[3]) / 255.0f;

    } else {
        clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
        clipCol[0] = clipCol[1] = clipCol[2] = 0.75f;
        clipCol[3] = 1.0f;
    }
}

bool OSPRayNHSphereGeometry::getExtends(megamol::core::Call &call) {
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