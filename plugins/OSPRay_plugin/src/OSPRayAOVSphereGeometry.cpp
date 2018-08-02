/*
 * OSPRayAOVSphereGeometry.cpp
 * Copyright (C) 2009-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "OSPRayAOVSphereGeometry.h"
#include "mmcore/Call.h"
#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "vislib/sys/Log.h"


using namespace megamol::ospray;


typedef float (*floatFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);
typedef unsigned char (*byteFromArrayFunc)(
    const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);


OSPRayAOVSphereGeometry::OSPRayAOVSphereGeometry(void)
    : particleList("ParticleList", "Switches between particle lists")
    , samplingRateSlot("samplingrate", "Set the samplingrate for the ao volume")
    , aoThresholdSlot(
          "aoThreshold", "Set the threshold for the ao vol sampling above which a sample is assumed to occlude")
    , getDataSlot("getdata", "Connects to the data source")
    , getVolSlot("getVol", "Connects to the density volume provider") {

    this->getDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getVolSlot.SetCompatibleCall<core::misc::VolumeticDataCallDescription>();
    this->MakeSlotAvailable(&this->getVolSlot);

    this->particleList << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->particleList);

    this->samplingRateSlot << new core::param::FloatParam(0.125f, 0.0f, std::numeric_limits<float>::max());
    this->MakeSlotAvailable(&this->samplingRateSlot);

    this->aoThresholdSlot << new core::param::FloatParam(1.0f, 0.0f, std::numeric_limits<float>::max());
    this->MakeSlotAvailable(&this->aoThresholdSlot);
}


bool OSPRayAOVSphereGeometry::readData(megamol::core::Call& call) {

    // fill material container
    this->processMaterial();

    // read Data, calculate  shape parameters, fill data vectors
    auto os = dynamic_cast<CallOSPRayStructure*>(&call);
    auto cd = this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();
    auto vd = this->getVolSlot.CallAs<core::misc::VolumetricDataCall>();
    if (vd == nullptr) return false;

    this->structureContainer.dataChanged = false;
    if (cd == nullptr) return false;
    cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    vd->SetFrameID(os->getTime(), true);
    if (this->datahash != cd->DataHash() || this->time != os->getTime() || this->volDatahash != vd->DataHash() ||
        this->volFrameID != vd->FrameID() || this->InterfaceIsDirty()) {
        this->datahash = cd->DataHash();
        this->time = os->getTime();
        this->volDatahash = vd->DataHash();
        this->volFrameID = os->getTime();
        this->structureContainer.dataChanged = true;
    } else {
        return true;
    }

    if (!(*cd)(1)) return false;
    if (!(*cd)(0)) return false;

    if (!(*vd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;
    if (!(*vd)(core::misc::VolumetricDataCall::IDX_GET_DATA)) return false;

    if (cd->GetParticleListCount() == 0) return false;

    if (this->particleList.Param<core::param::IntParam>()->Value() > (cd->GetParticleListCount() - 1)) {
        this->particleList.Param<core::param::IntParam>()->SetValue(0);
    }

    core::moldyn::MultiParticleDataCall::Particles& parts =
        cd->AccessParticles(this->particleList.Param<core::param::IntParam>()->Value());

    unsigned int const partCount = parts.GetCount();
    float const globalRadius = parts.GetGlobalRadius();

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

    int vstride;
    if (parts.GetVertexDataStride() == 0) {
        vstride = core::moldyn::MultiParticleDataCall::Particles::VertexDataSize[parts.GetVertexDataType()];
    }

    if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE &&
        parts.GetColourDataType() != core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE) {
        vislib::sys::Log::DefaultLog.WriteError("Only color data is not allowed.");
    }

    // get the volume stuff
    auto const volSDT = vd->GetScalarType(); //< unfortunately only float is part of the intersection
    if (volSDT != core::misc::VolumetricDataCall::ScalarType::FLOATING_POINT) {
        vislib::sys::Log::DefaultLog.WriteError(
            "OSPRayAOVSphereGeometry: Only float is supported as AOVol data type\n");
        return false;
    }
    auto const volGT = vd->GetGridType();
    if (volGT != core::misc::VolumetricDataCall::GridType::CARTESIAN &&
        volGT != core::misc::VolumetricDataCall::GridType::RECTILINEAR) {
        vislib::sys::Log::DefaultLog.WriteError(
            "OSPRayAOVSphereGeometry: Currently only grids are supported as AOVol grid type\n");
        return false;
    }
    auto const volCom = vd->GetComponents();
    if (volCom != 1) {
        vislib::sys::Log::DefaultLog.WriteError(
            "OSPRayAOVSphereGeometry: Only one component per cell is allowed as AOVol\n");
        return false;
    }
    auto const metadata = vd->GetMetadata();
    if (metadata->MinValues == nullptr || metadata->MaxValues == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("OSPRayAOVSphereGeometry: AOVol requires a specified value range\n");
        return false;
    }
    float const minV = metadata->MinValues[0];
    float const maxV = metadata->MaxValues[0];
    this->valuerange = std::make_pair(minV, maxV);
    this->gridorigin = {metadata->Origin[0], metadata->Origin[1], metadata->Origin[2]};
    this->gridspacing = {metadata->SliceDists[0][0], metadata->SliceDists[1][0], metadata->SliceDists[2][0]};
    this->dimensions = {static_cast<int>(metadata->Resolution[0]), static_cast<int>(metadata->Resolution[1]),
        static_cast<int>(metadata->Resolution[2])}; //< TODO HAZARD explizit narrowing

    // Write stuff into the structureContainer
    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::AOVSPHERES;
    this->structureContainer.raw = std::make_shared<const void*>(parts.GetVertexData());
    this->structureContainer.vertexLength = vertexLength;
    this->structureContainer.vertexStride = vstride;
    this->structureContainer.colorLength = colorLength;
    this->structureContainer.colorStride = parts.GetColourDataStride();
    this->structureContainer.partCount = partCount;
    this->structureContainer.globalRadius = globalRadius;
    this->structureContainer.mmpldColor = parts.GetColourDataType();

    this->structureContainer.raw2 = std::make_shared<void const*>(vd->GetData());
    this->structureContainer.valueRange =
        std::make_shared<std::pair<float, float>>(this->valuerange); //< TODO HAZARD potential dangling shared pointer
    this->structureContainer.gridOrigin = std::make_shared<std::vector<float>>(this->gridorigin);   //<
    this->structureContainer.gridSpacing = std::make_shared<std::vector<float>>(this->gridspacing); //<
    this->structureContainer.dimensions = std::make_shared<std::vector<int>>(this->dimensions);     //<
    this->structureContainer.voxelDType = voxelDataType::FLOAT;
    this->structureContainer.samplingRate = this->samplingRateSlot.Param<core::param::FloatParam>()->Value();
    this->structureContainer.aoThreshold = this->aoThresholdSlot.Param<core::param::FloatParam>()->Value();

    return true;
}


OSPRayAOVSphereGeometry::~OSPRayAOVSphereGeometry() { this->Release(); }


bool OSPRayAOVSphereGeometry::create() { return true; }


void OSPRayAOVSphereGeometry::release() {}


bool OSPRayAOVSphereGeometry::InterfaceIsDirty() {
    if (this->particleList.IsDirty()) {
        this->particleList.ResetDirty();
        return true;
    } else {
        return false;
    }
}


bool OSPRayAOVSphereGeometry::getExtends(megamol::core::Call& call) {
    auto os = dynamic_cast<CallOSPRayStructure*>(&call);
    auto cd = this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();

    if (cd == nullptr) return false;
    cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    // if (!(*cd)(1)) return false; // floattable returns flase at first attempt and breaks everything
    (*cd)(1);
    this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes>(cd->AccessBoundingBoxes());
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}