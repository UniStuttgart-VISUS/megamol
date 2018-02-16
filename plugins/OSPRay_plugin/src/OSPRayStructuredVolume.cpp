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
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/EnumParam.h"
#include "vislib/sys/Log.h"
#include "mmcore/Call.h"
#include "mmcore/view/CallGetTransferFunction.h"


using namespace megamol::ospray;


OSPRayStructuredVolume::OSPRayStructuredVolume(void) :
    AbstractOSPRayStructure(),

    clippingBoxActive("ClippingBox::Active", "Activates the clipping Box"),
    clippingBoxLower("ClippingBox::Left", "Left corner of the clipping Box"),
    clippingBoxUpper("ClippingBox::Right", "Right corner of the clipping Box"),
    repType("Representation", "Activates one of the three different volume representations: Volume, Isosurfae, Slice"),
    sliceNormal("Slice::sliceNormal", "Direction of the slice normal"),
    sliceDist("Slice::sliceDist", "Distance of the slice in the direction of the normal vector"),
    IsoValue("Isosurface::Isovalue","Sets the isovalue of the isosurface"),
    getTFSlot("gettransferfunction", "Connects to a color transfer function module"),

    getDataSlot("getdata", "Connects to the data source")
 {
    core::param::EnumParam *rt = new core::param::EnumParam(VOLUMEREP);
    rt->SetTypePair(VOLUMEREP, "Volume");
    rt->SetTypePair(ISOSURFACE, "Isosurface");
    rt->SetTypePair(SLICE, "Slice");
    this->repType << rt;
    this->MakeSlotAvailable(&this->repType);

    this->clippingBoxActive << new core::param::BoolParam(false);
    this->clippingBoxLower << new core::param::Vector3fParam({ -5.0f, -5.0f, -5.0f });
    this->clippingBoxUpper << new core::param::Vector3fParam({ 0.0f, 5.0f, 5.0f });
    this->MakeSlotAvailable(&this->clippingBoxActive);
    this->MakeSlotAvailable(&this->clippingBoxLower);
    this->MakeSlotAvailable(&this->clippingBoxUpper);

    this->sliceNormal << new core::param::Vector3fParam({ 1.0f, 0.0f, 0.0f });
    this->sliceDist << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->sliceNormal);
    this->MakeSlotAvailable(&this->sliceDist);

    this->IsoValue << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->IsoValue);

    this->getDataSlot.SetCompatibleCall<megamol::core::moldyn::VolumeDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    //this->SetSlotUnavailable(&this->getMaterialSlot);
}


bool OSPRayStructuredVolume::readData(megamol::core::Call &call) {

    // fill material container
    this->processMaterial();

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

    // get color transfer function
    std::vector<float> rgb;
    std::vector<float> a;
    core::view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<core::view::CallGetTransferFunction>();
    if (cgtf != NULL && ((*cgtf)())) {
        if (cgtf->OpenGLTextureFormat() == megamol::core::view::CallGetTransferFunction::TextureFormat::TEXTURE_FORMAT_RGBA) {
            auto numColors = cgtf->TextureSize() / 4;
            rgb.resize(3 * numColors);
            a.resize(numColors);
            auto texture = cgtf->GetTextureData();

            for (unsigned int i = 0; i < numColors; i++) {
                rgb[i + 0] = texture[i + 0];
                rgb[i + 1] = texture[i + 1];
                rgb[i + 2] = texture[i + 2];
                a[i]       = texture[i + 3];
            }
        } else {
            vislib::sys::Log::DefaultLog.WriteError("No color transfer function connected to OSPRayStructuredVolume module");
            return false;
        }
    }



    // Write stuff into the structureContainer

    this->structureContainer.type = structureTypeEnum::VOLUME;
    this->structureContainer.volumeType = volumeTypeEnum::STRUCTUREDVOLUME;
    this->structureContainer.volRepType = (volumeRepresentationType)this->repType.Param<core::param::EnumParam>()->Value();
    this->structureContainer.voxels = std::make_shared<std::vector<float>>(std::move(voxels));
    this->structureContainer.gridOrigin = std::make_shared<std::vector<float>>(std::move(gridOrigin));
    this->structureContainer.gridSpacing = std::make_shared<std::vector<float>>(std::move(gridSpacing));
    this->structureContainer.dimensions = std::make_shared<std::vector<int>>(std::move(dimensions));
    this->structureContainer.voxelCount = voxelCount;
    this->structureContainer.maxDim = maxDim;
    this->structureContainer.tfRGB = std::make_shared<std::vector<float>>(std::move(rgb));
    this->structureContainer.tfA = std::make_shared<std::vector<float>>(std::move(a));


    this->structureContainer.clippingBoxActive = this->clippingBoxActive.Param<core::param::BoolParam>()->Value();
    std::vector<float> cbl = { this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetX(), this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetY(), this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetZ() };
    this->structureContainer.clippingBoxLower = std::make_shared<std::vector<float>>(std::move(cbl));
    std::vector<float> cbu = { this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetX(), this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetY(), this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetZ() };
    this->structureContainer.clippingBoxUpper = std::make_shared<std::vector<float>>(std::move(cbu));

    std::vector<float> sData = { this->sliceNormal.Param<core::param::Vector3fParam>()->Value().GetX(), this->sliceNormal.Param<core::param::Vector3fParam>()->Value().GetY(), this->sliceNormal.Param<core::param::Vector3fParam>()->Value().GetZ(), this->sliceDist.Param<core::param::FloatParam>()->Value() };
    this->structureContainer.sliceData = std::make_shared<std::vector<float>>(std::move(sData));

    std::vector<float> iValue = { this->IsoValue.Param<core::param::FloatParam>()->Value() };
    this->structureContainer.isoValue = std::make_shared<std::vector<float>>(std::move(iValue));

    return true;
}


OSPRayStructuredVolume::~OSPRayStructuredVolume() {
    this->Release();
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
    if (
        this->clippingBoxActive.IsDirty() ||
        this->clippingBoxLower.IsDirty() ||
        this->clippingBoxUpper.IsDirty() || 
        this->sliceDist.IsDirty() || 
        this->sliceNormal.IsDirty() ||
        this->IsoValue.IsDirty() ||
        this->repType.IsDirty()
        ) {
        this->clippingBoxActive.ResetDirty();
        this->clippingBoxLower.ResetDirty();
        this->clippingBoxUpper.ResetDirty();
        this->sliceDist.ResetDirty();
        this->sliceNormal.ResetDirty();
        this->IsoValue.ResetDirty();
        this->repType.ResetDirty();
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