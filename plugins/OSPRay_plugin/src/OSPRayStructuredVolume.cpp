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
#include "mmcore/param/StringParam.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/utility/log/Log.h"


using namespace megamol::ospray;


OSPRayStructuredVolume::OSPRayStructuredVolume(void)
    : AbstractOSPRayStructure()
    , getDataSlot("getdata", "Connects to the data source")
    , getTFSlot("gettransferfunction", "Connects to a color transfer function module")
    , clippingBoxLower("ClippingBox::Left", "Left corner of the clipping Box")
    , clippingBoxUpper("ClippingBox::Right", "Right corner of the clipping Box")
    , clippingBoxActive("ClippingBox::Active", "Activates the clipping Box")
    , repType("Representation", "Activates one of the three different volume representations: Volume, Isosurfae, Slice")
    , IsoValue("Isosurface::Isovalue", "Sets the isovalue of the isosurface")
    , showBoundingBox("showBoundingBox", "Bounding box of the volume data set") {
    core::param::EnumParam* rt = new core::param::EnumParam(VOLUMEREP);
    rt->SetTypePair(VOLUMEREP, "Volume");
    rt->SetTypePair(ISOSURFACE, "Isosurface");
    this->repType << rt;
    this->MakeSlotAvailable(&this->repType);

    this->clippingBoxActive << new core::param::BoolParam(false);
    this->clippingBoxLower << new core::param::Vector3fParam({-5.0f, -5.0f, -5.0f});
    this->clippingBoxUpper << new core::param::Vector3fParam({0.0f, 5.0f, 5.0f});
    this->MakeSlotAvailable(&this->clippingBoxActive);
    this->MakeSlotAvailable(&this->clippingBoxLower);
    this->MakeSlotAvailable(&this->clippingBoxUpper);

    this->IsoValue << new core::param::FloatParam(0.1f);
    this->MakeSlotAvailable(&this->IsoValue);

    this->getDataSlot.SetCompatibleCall<core::misc::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);
    
    this->showBoundingBox << new core::param::StringParam("");
    this->showBoundingBox.Parameter()->SetGUIReadOnly(true);
    this->MakeSlotAvailable(&this->showBoundingBox);

    // this->SetSlotUnavailable(&this->getMaterialSlot);
}


bool OSPRayStructuredVolume::readData(megamol::core::Call& call) {

    // fill material container
    this->processMaterial();

    // get transformation parameter
    //this->processTransformation();

    // fill clipping plane container
    this->processClippingPlane();

    // read Data, calculate  shape parameters, fill data vectors
    auto os = dynamic_cast<CallOSPRayStructure*>(&call);
    auto cd = this->getDataSlot.CallAs<megamol::core::misc::VolumetricDataCall>();
    auto const cgtf = this->getTFSlot.CallAs<core::view::CallGetTransferFunction>();

    this->structureContainer.dataChanged = false;
    if (cd == nullptr) return false;
    if (cgtf == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[OSPRayStructuredVolume] No transferfunction connected.");
        return false;
    }

    uint32_t requested_frame = os->getTime();
    if (requested_frame >= cd->FrameCount()) {
        requested_frame = cd->FrameCount() - 1;
    }
    do {
        cd->SetFrameID(requested_frame, true);
        if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;
        if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) return false;
        if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_DATA)) return false;
    } while (cd->FrameID() != requested_frame);

    // do the callback to set the dirty flag
    if (!(*cgtf)(0)) return false;
    auto interface_diry = this->InterfaceIsDirty();
    auto tf_dirty = cgtf->IsDirty();
    if (this->datahash != cd->DataHash() || this->time != os->getTime() || interface_diry || tf_dirty) {
        this->datahash = cd->DataHash();
        this->time = os->getTime();
        this->structureContainer.dataChanged = true;
    } else {
        return true;
    }

    auto const metadata = cd->GetMetadata();

    if (metadata->GridType != core::misc::CARTESIAN) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("OSPRayStructuredVolume only works with cartesian grids (for now)");
        return false;
    }

    unsigned int const voxelCount = metadata->Resolution[0] * metadata->Resolution[1] * metadata->Resolution[2];
    std::array<float,3> gridOrigin = {metadata->Origin[0], metadata->Origin[1], metadata->Origin[2]};
    std::array<float,3> gridSpacing = {
        metadata->SliceDists[0][0], metadata->SliceDists[1][0], metadata->SliceDists[2][0]};
    std::array<int, 3> dimensions = {static_cast<int>(metadata->Resolution[0]), static_cast<int>(metadata->Resolution[1]),
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
            megamol::core::utility::log::Log::DefaultLog.WriteError("Unsigned integers with a length greater than 2 are invalid.");
            return false;
        }
        break;
    case core::misc::SIGNED_INTEGER:
        if (metadata->ScalarLength == 2) {
            voxelType = voxelDataType::SHORT;
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError("Integers with a length != 2 are invalid.");
            return false;
        }
        break;
    case core::misc::BITS:
        megamol::core::utility::log::Log::DefaultLog.WriteError("Invalid datatype.");
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
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("OSPRayStructuredVolume: No alpha channel in transfer function "
                                                   "connected to module. Adding alpha ramp to RGB colors.\n");
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError("OSPRayStructuredVolume: No transfer function connected to module");
        return false;
    }
    cgtf->ResetDirty();

    // Write stuff into the structureContainer

    this->structureContainer.type = structureTypeEnum::VOLUME;
    this->structureContainer.volumeType = volumeTypeEnum::STRUCTUREDVOLUME;
    structuredVolumeStructure svs;

    svs.volRepType =
        (volumeRepresentationType)this->repType.Param<core::param::EnumParam>()->Value();
    svs.voxels = cd->GetData();
    svs.gridOrigin = gridOrigin;
    svs.gridSpacing = gridSpacing;
    svs.dimensions = dimensions;
    svs.voxelCount = voxelCount;
    svs.maxDim = maxDim;
    svs.valueRange = {static_cast<float>(metadata->MinValues[0]), static_cast<float>(metadata->MaxValues[0])};
    svs.tfRGB = std::make_shared<std::vector<float>>(std::move(rgb));
    svs.tfA = std::make_shared<std::vector<float>>(std::move(a));
    svs.voxelDType = voxelType;

    svs.clippingBoxActive = this->clippingBoxActive.Param<core::param::BoolParam>()->Value();
    std::array<float,3> cbl = {this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetX(),
        this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetY(),
        this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetZ()};
    svs.clippingBoxLower = cbl;
    std::array<float,3> cbu = {this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetX(),
        this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetY(),
        this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetZ()};
    svs.clippingBoxUpper = cbu;

    svs.isoValue = this->IsoValue.Param<core::param::FloatParam>()->Value();

    this->structureContainer.structure = svs;

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
        this->IsoValue.IsDirty() ||
        this->repType.IsDirty()) {
        this->clippingBoxActive.ResetDirty();
        this->clippingBoxLower.ResetDirty();
        this->clippingBoxUpper.ResetDirty();
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
    std::string bbox_string =
                              "LEFT: " + std::to_string(extendContainer.boundingBox->BoundingBox().Left()) +
                              ";\nBOTTOM: " + std::to_string(extendContainer.boundingBox->BoundingBox().Bottom()) +
                              ";\nBACK: " + std::to_string(extendContainer.boundingBox->BoundingBox().Back()) +
                              ";\nRIGHT: " + std::to_string(extendContainer.boundingBox->BoundingBox().Right()) +
                              ";\nTOP: " + std::to_string(extendContainer.boundingBox->BoundingBox().Top()) +
                              ";\nFRONT: " + std::to_string(extendContainer.boundingBox->BoundingBox().Bottom());
    this->showBoundingBox.Param<core::param::StringParam>()->SetValue(bbox_string.c_str());
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}
