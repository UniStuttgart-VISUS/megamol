/*
 * OSPRayStructuredVolume.cpp
 * Copyright (C) 2009-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "OSPRayStructuredVolume.h"
#include "mmcore/Call.h"
#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "vislib/sys/Log.h"


using namespace megamol::ospray;


OSPRayStructuredVolume::OSPRayStructuredVolume(void)
    : AbstractOSPRayStructure()
    , getDataSlot("getdata", "Connects to the data source")
    , getTFSlot("gettransferfunction", "Connects to a color transfer function module")
    , clippingBoxLower("ClippingBox::Left", "Left corner of the clipping Box")
    , clippingBoxUpper("ClippingBox::Right", "Right corner of the clipping Box")
    , clippingBoxActive("ClippingBox::Active", "Activates the clipping Box")
    , repType("Representation", "Activates one of the three different volume representations: Volume, Isosurfae, Slice")
    , useMIP("shading::useMIP", "toggle maximum intensity projection")
    , useGradient("shading::useGradient", "compute gradient for shading")
    , usePreIntegration("shading::usePreintegration", "toggle preintegration")
    , useAdaptiveSampling("adaptive::enable", "toggle adaptive sampling")
    , adaptiveFactor("adaptive::factor", "modifier for adaptive step size")
    , adaptiveMaxRate("adaptive::maxRate", "maximum sampling rate")
    , samplingRate("adaptive::minRate", "minimum sampling rate")
    , IsoValue("Isosurface::Isovalue", "Sets the isovalue of the isosurface")
    , sliceNormal("Slice::sliceNormal", "Direction of the slice normal")
    , sliceDist("Slice::sliceDist", "Distance of the slice in the direction of the normal vector") {
    core::param::EnumParam* rt = new core::param::EnumParam(VOLUMEREP);
    rt->SetTypePair(VOLUMEREP, "Volume");
    rt->SetTypePair(ISOSURFACE, "Isosurface");
    rt->SetTypePair(SLICE, "Slice");
    this->repType << rt;
    this->MakeSlotAvailable(&this->repType);

    this->useMIP << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->useMIP);
    this->useGradient << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->useGradient);
    this->usePreIntegration << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->usePreIntegration);

    this->useAdaptiveSampling << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->useAdaptiveSampling);
    this->adaptiveFactor << new core::param::FloatParam(15.0f, std::numeric_limits<float>::min());
    this->MakeSlotAvailable(&this->adaptiveFactor);
    this->adaptiveMaxRate << new core::param::FloatParam(2.0f, std::numeric_limits<float>::min());
    this->MakeSlotAvailable(&this->adaptiveMaxRate);
    this->samplingRate << new core::param::FloatParam(0.125f, std::numeric_limits<float>::min());
    this->MakeSlotAvailable(&this->samplingRate);

    this->clippingBoxActive << new core::param::BoolParam(false);
    this->clippingBoxLower << new core::param::Vector3fParam({-5.0f, -5.0f, -5.0f});
    this->clippingBoxUpper << new core::param::Vector3fParam({0.0f, 5.0f, 5.0f});
    this->MakeSlotAvailable(&this->clippingBoxActive);
    this->MakeSlotAvailable(&this->clippingBoxLower);
    this->MakeSlotAvailable(&this->clippingBoxUpper);

    this->sliceNormal << new core::param::Vector3fParam({1.0f, 0.0f, 0.0f});
    this->sliceDist << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->sliceNormal);
    this->MakeSlotAvailable(&this->sliceDist);

    this->IsoValue << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->IsoValue);

    /*this->getDataSlot.SetCompatibleCall<megamol::core::moldyn::VolumeDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);*/
    this->getDataSlot.SetCompatibleCall<core::misc::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    // this->SetSlotUnavailable(&this->getMaterialSlot);
}


bool OSPRayStructuredVolume::readData(megamol::core::Call& call) {

    // fill material container
    this->processMaterial();

    // get transformation parameter
    //this->processTransformation();

    // read Data, calculate  shape parameters, fill data vectors
    auto os = dynamic_cast<CallOSPRayStructure*>(&call);
    auto cd = this->getDataSlot.CallAs<megamol::core::misc::VolumetricDataCall>();
    auto const cgtf = this->getTFSlot.CallAs<core::view::CallGetTransferFunction>();

    this->structureContainer.dataChanged = false;
    if (cd == nullptr) return false;
    if (cgtf == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[OSPRayStructuredVolume] No transferfunction connected.");
        return false;
    }

    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;
    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) return false;
    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_DATA)) return false;

    if (os->getTime() >= cd->FrameCount()) {
        cd->SetFrameID(cd->FrameCount() - 1, true); // isTimeForced flag set to true
    } else {
        cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    }
    
    // do the callback to set the dirty flag
    if (!(*cgtf)(0)) return false;
    
    if (this->datahash != cd->DataHash() || this->time != os->getTime() || this->InterfaceIsDirty() ||
        cgtf->IsDirty()) {
        this->datahash = cd->DataHash();
        this->time = os->getTime();
        this->structureContainer.dataChanged = true;
    } else {
        return true;
    }

    auto const metadata = cd->GetMetadata();

    if (!metadata->GridType == core::misc::CARTESIAN) {
        vislib::sys::Log::DefaultLog.WriteError("OSPRayStructuredVolume only works with cartesian grids (for now)");
        return false;
    }

    /*unsigned int voxelCount =
        cd->VolumeDimension().GetDepth() * cd->VolumeDimension().GetHeight() * cd->VolumeDimension().GetWidth();*/
    unsigned int const voxelCount = metadata->Resolution[0] * metadata->Resolution[1] * metadata->Resolution[2];

    /*unsigned int maxDim = vislib::math::Max<unsigned int>(cd->VolumeDimension().Depth(),
        vislib::math::Max<unsigned int>(cd->VolumeDimension().Height(), cd->VolumeDimension().Width()));*/

    // float scale = 2.0f;
    /*std::vector<float> gridOrigin = {-0.5f * scale, -0.5f * scale, -0.5f * scale};
    std::vector<float> gridSpacing = {scale / (float)maxDim, scale / (float)maxDim, scale / (float)maxDim};
    std::vector<int> dimensions = {(int)cd->VolumeDimension().GetWidth(), (int)cd->VolumeDimension().GetHeight(),
        (int)cd->VolumeDimension().GetDepth()};*/

    std::vector<float> gridOrigin = {metadata->Origin[0], metadata->Origin[1], metadata->Origin[2]};
    std::vector<float> gridSpacing = {
        metadata->SliceDists[0][0], metadata->SliceDists[1][0], metadata->SliceDists[2][0]};
    std::vector<int> dimensions = {static_cast<int>(metadata->Resolution[0]), static_cast<int>(metadata->Resolution[1]),
        static_cast<int>(metadata->Resolution[2])}; //< TODO HAZARD explicit narrowing

    unsigned int const maxDim =
        std::max<size_t>(metadata->Resolution[0], std::max<size_t>(metadata->Resolution[1], metadata->Resolution[2]));

    voxelDataType voxelType = {};

    switch (metadata->ScalarType) {
    case core::misc::FLOATING_POINT:
        if (metadata->ScalarLength == 4) {
            voxelType = voxelDataType::FLOAT;
        } else {
            voxelType = voxelDataType::DOUBLE;
        }
        break;
    case core::misc::UNSIGNED_INTEGER:
        if (metadata->ScalarLength == 1) {
            voxelType = voxelDataType::UCHAR;
        } else if (metadata->ScalarLength == 2) {
            voxelType = voxelDataType::USHORT;
        } else {
            vislib::sys::Log::DefaultLog.WriteError("Unsigned integers with a length greater than 2 are invalid.");
            return false;
        }
        break;
    case core::misc::SIGNED_INTEGER:
        if (metadata->ScalarLength == 2) {
            voxelType = voxelDataType::SHORT;
        } else {
            vislib::sys::Log::DefaultLog.WriteError("Integers with a length != 2 are invalid.");
            return false;
        }
        break;
    case core::misc::BITS:
        vislib::sys::Log::DefaultLog.WriteError("Invalid datatype.");
        return false;
        break;
    }

    // get color transfer function
    std::vector<float> rgb;
    std::vector<float> a;

    if ((*cgtf)(0)) {
        if (cgtf->OpenGLTextureFormat() ==
            megamol::core::view::CallGetTransferFunction::TextureFormat::TEXTURE_FORMAT_RGBA) {
            auto const numColors = cgtf->TextureSize();
            rgb.resize(3 * numColors);
            a.resize(numColors);
            auto const texture = cgtf->GetTextureData();

            for (unsigned int i = 0; i < numColors; ++i) {
                rgb[i * 3 + 0] = texture[i * 4 + 0];
                rgb[i * 3 + 1] = texture[i * 4 + 1];
                rgb[i * 3 + 2] = texture[i * 4 + 2];
                a[i] = texture[i * 4 + 3];
            }
        } else {
            auto const numColors = cgtf->TextureSize();
            rgb.resize(3 * numColors);
            a.resize(numColors);
            auto const texture = cgtf->GetTextureData();

            for (unsigned int i = 0; i < numColors; ++i) {
                rgb[i * 3 + 0] = texture[i * 4 + 0];
                rgb[i * 3 + 1] = texture[i * 4 + 1];
                rgb[i * 3 + 2] = texture[i * 4 + 2];
                a[i] = i / (numColors - 1.0f);
            }
            vislib::sys::Log::DefaultLog.WriteWarn("OSPRayStructuredVolume: No alpha channel in transfer function "
                                                   "connected to module. Adding alpha ramp to RGB colors.\n");
        }
    } else {
        vislib::sys::Log::DefaultLog.WriteError("OSPRayStructuredVolume: No transfer function connected to module");
        return false;
    }
    cgtf->ResetDirty();

    // Write stuff into the structureContainer

    this->structureContainer.type = structureTypeEnum::VOLUME;
    this->structureContainer.volumeType = volumeTypeEnum::STRUCTUREDVOLUME;
    this->structureContainer.volRepType =
        (volumeRepresentationType)this->repType.Param<core::param::EnumParam>()->Value();
    this->structureContainer.voxels = cd->GetData();
    this->structureContainer.gridOrigin = std::make_shared<std::vector<float>>(std::move(gridOrigin));
    this->structureContainer.gridSpacing = std::make_shared<std::vector<float>>(std::move(gridSpacing));
    this->structureContainer.dimensions = std::make_shared<std::vector<int>>(std::move(dimensions));
    this->structureContainer.voxelCount = voxelCount;
    this->structureContainer.maxDim = maxDim;
    this->structureContainer.valueRange =
        std::make_shared<std::pair<float, float>>(metadata->MinValues[0], metadata->MaxValues[0]);
    this->structureContainer.tfRGB = std::make_shared<std::vector<float>>(std::move(rgb));
    this->structureContainer.tfA = std::make_shared<std::vector<float>>(std::move(a));
    this->structureContainer.voxelDType = voxelType;

    this->structureContainer.clippingBoxActive = this->clippingBoxActive.Param<core::param::BoolParam>()->Value();
    std::vector<float> cbl = {this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetX(),
        this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetY(),
        this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetZ()};
    this->structureContainer.clippingBoxLower = std::make_shared<std::vector<float>>(std::move(cbl));
    std::vector<float> cbu = {this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetX(),
        this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetY(),
        this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetZ()};
    this->structureContainer.clippingBoxUpper = std::make_shared<std::vector<float>>(std::move(cbu));

    std::vector<float> sData = {this->sliceNormal.Param<core::param::Vector3fParam>()->Value().GetX(),
        this->sliceNormal.Param<core::param::Vector3fParam>()->Value().GetY(),
        this->sliceNormal.Param<core::param::Vector3fParam>()->Value().GetZ(),
        this->sliceDist.Param<core::param::FloatParam>()->Value()};
    this->structureContainer.sliceData = std::make_shared<std::vector<float>>(std::move(sData));

    std::vector<float> iValue = {this->IsoValue.Param<core::param::FloatParam>()->Value()};
    this->structureContainer.isoValue = std::make_shared<std::vector<float>>(std::move(iValue));

    this->structureContainer.useMIP = this->useMIP.Param<core::param::BoolParam>()->Value();
    this->structureContainer.useAdaptiveSampling = this->useAdaptiveSampling.Param<core::param::BoolParam>()->Value();
    this->structureContainer.useGradient = this->useGradient.Param<core::param::BoolParam>()->Value();
    this->structureContainer.usePreIntegration = this->usePreIntegration.Param<core::param::BoolParam>()->Value();
    this->structureContainer.adaptiveFactor = this->adaptiveFactor.Param<core::param::FloatParam>()->Value();
    this->structureContainer.adaptiveMaxRate = this->adaptiveMaxRate.Param<core::param::FloatParam>()->Value();
    this->structureContainer.samplingRate = this->samplingRate.Param<core::param::FloatParam>()->Value();

    return true;
}


OSPRayStructuredVolume::~OSPRayStructuredVolume() { this->Release(); }

bool OSPRayStructuredVolume::create() { return true; }

void OSPRayStructuredVolume::release() {}

/*
ospray::OSPRaySphereGeometry::InterfaceIsDirty()
*/
bool OSPRayStructuredVolume::InterfaceIsDirty() {
    if (this->clippingBoxActive.IsDirty() || this->clippingBoxLower.IsDirty() || this->clippingBoxUpper.IsDirty() ||
        this->sliceDist.IsDirty() || this->sliceNormal.IsDirty() || this->IsoValue.IsDirty() ||
        this->repType.IsDirty()) {
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


bool OSPRayStructuredVolume::getExtends(megamol::core::Call& call) {
    auto os = dynamic_cast<CallOSPRayStructure*>(&call);
    auto cd = this->getDataSlot.CallAs<megamol::core::misc::VolumetricDataCall>();

    if (cd == nullptr) return false;
    if (os->getTime() > cd->FrameCount()) {
        cd->SetFrameID(cd->FrameCount() - 1, true); // isTimeForced flag set to true
    } else {
        cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    }

    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;
    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) return false;

    this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes_2>();
    this->extendContainer.boundingBox->SetBoundingBox(cd->AccessBoundingBoxes().ObjectSpaceBBox());
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}