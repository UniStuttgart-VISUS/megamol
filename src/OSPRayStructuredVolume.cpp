/*
* OSPRayStructuredVolume.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayStructuredVolume.h"
#include "mmcore/moldyn/VolumeDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "vislib/sys/Log.h"
#include "mmcore/Call.h"


using namespace megamol::ospray;


OSPRayStructuredVolume::OSPRayStructuredVolume(void) :
    AbstractOSPRayStructure(),

    clippingBoxActive("Volume::ClippingBox::Active", "Activates the clipping Box"),
    clippingBoxLower("Volume::ClippingBox::Left", "Left corner of the clipping Box"),
    clippingBoxUpper("Volume::ClippingBox::Right", "Right corner of the clipping Box"),

    getDataSlot("getdata", "Connects to the data source")
 {

    this->clippingBoxActive << new core::param::BoolParam(false);
    this->clippingBoxLower << new core::param::Vector3fParam({ -5.0f, -5.0f, -5.0f });
    this->clippingBoxUpper << new core::param::Vector3fParam({ 0.0f, 5.0f, 5.0f });
    this->MakeSlotAvailable(&this->clippingBoxActive);
    this->MakeSlotAvailable(&this->clippingBoxLower);
    this->MakeSlotAvailable(&this->clippingBoxUpper);

    this->getDataSlot.SetCompatibleCall<megamol::core::moldyn::VolumeDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    //this->SetSlotUnavailable(&this->getMaterialSlot);

}


bool OSPRayStructuredVolume::readData(megamol::core::Call &call) {

    // read Data, calculate  shape parameters, fill data vectors

    CallOSPRayStructure *os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::core::moldyn::VolumeDataCall *cd = this->getDataSlot.CallAs<megamol::core::moldyn::VolumeDataCall>();

    this->structureContainer.dataChanged = false;
    if (cd == NULL) return false;
    if (os->getTime() > cd->FrameCount()) {
        cd->SetFrameID(cd->FrameCount() - 1, true); // isTimeForced flag set to true
    } else {
        cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    }
    if (this->datahash != cd->DataHash() || this->time != os->getTime() || this->InterfaceIsDirty()) {
        this->datahash = cd->DataHash();
        this->time = os->getTime();
        this->structureContainer.dataChanged = true;
    } else {
        return true;
    }


    if (!(*cd)(1)) return false;
    if (!(*cd)(0)) return false;


    unsigned int voxelCount = cd->VolumeDimension().GetDepth() * cd->VolumeDimension().GetHeight() * cd->VolumeDimension().GetWidth();
    unsigned int maxDim = vislib::math::Max<unsigned int>(cd->VolumeDimension().Depth(), vislib::math::Max<unsigned int>(cd->VolumeDimension().Height(), cd->VolumeDimension().Width()));
    float scale = 2.0f;
    std::vector<float> gridOrigin = { -0.5f*scale, -0.5f*scale, -0.5f*scale };
    std::vector<float> gridSpacing = { scale / (float)maxDim, scale / (float)maxDim, scale / (float)maxDim };
    std::vector<int> dimensions = { (int)cd->VolumeDimension().GetWidth(), (int)cd->VolumeDimension().GetHeight(), (int)cd->VolumeDimension().GetDepth() };


    std::vector<float> voxels(voxelCount);
    voxels.assign(cd->VoxelMap(), cd->VoxelMap() + voxelCount);


    // Write stuff into the structureContainer

    this->structureContainer.type = structureTypeEnum::VOLUME;
    this->structureContainer.voxels = std::make_shared<std::vector<float>>(std::move(voxels));
    this->structureContainer.gridOrigin = std::make_shared<std::vector<float>>(std::move(gridOrigin));
    this->structureContainer.gridSpacing = std::make_shared<std::vector<float>>(std::move(gridSpacing));
    this->structureContainer.dimensions = std::make_shared<std::vector<int>>(std::move(dimensions));
    this->structureContainer.voxelCount = voxelCount;
    this->structureContainer.maxDim = maxDim;

    this->structureContainer.clippingBoxActive = this->clippingBoxActive.Param<core::param::BoolParam>()->Value();
    std::vector<float> cbl = { this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetX(), this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetY(), this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetZ() };
    this->structureContainer.clippingBoxLower = std::make_shared<std::vector<float>>(std::move(cbl));
    std::vector<float> cbu = { this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetX(), this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetY(), this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetZ() };
    this->structureContainer.clippingBoxUpper = std::make_shared<std::vector<float>>(std::move(cbu));


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


OSPRayStructuredVolume::~OSPRayStructuredVolume() {
    //
}

bool OSPRayStructuredVolume::create() {
    return true;
}

void OSPRayStructuredVolume::release() {

}

/*
ospray::OSPRaySphereGeometry::InterfaceIsDirty()
*/
bool OSPRayStructuredVolume::InterfaceIsDirty() {
    CallOSPRayMaterial *cm = this->getMaterialSlot.CallAs<CallOSPRayMaterial>();
    cm->getMaterialParameter();
    if (
        cm->InterfaceIsDirty() ||
        this->clippingBoxActive.IsDirty() ||
        this->clippingBoxLower.IsDirty() ||
        this->clippingBoxUpper.IsDirty()
        ) {
        this->clippingBoxActive.ResetDirty();
        this->clippingBoxLower.ResetDirty();
        this->clippingBoxUpper.ResetDirty();
        return true;
    } else {
        return false;
    }
}



bool OSPRayStructuredVolume::getExtends(megamol::core::Call &call) {
    CallOSPRayStructure *os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::core::moldyn::VolumeDataCall *cd = this->getDataSlot.CallAs<megamol::core::moldyn::VolumeDataCall>();

    if (cd == NULL) return false;
    if (os->getTime() > cd->FrameCount()) {
        cd->SetFrameID(cd->FrameCount() - 1, true);  // isTimeForced flag set to true
    } else {
        cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    }

    if (!(*cd)(1)) return false;

    this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes>(cd->AccessBoundingBoxes());
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}