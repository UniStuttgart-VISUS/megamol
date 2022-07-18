/*
 * OSPRaySphericalVolume.cpp
 * Copyright (C) 2022 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "OSPRaySphericalVolume.h"
#include "mmcore/Call.h"
#include "mmadios/CallADIOSData.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/CallGetTransferFunction.h"


using namespace megamol::ospray;


OSPRaySphericalVolume::OSPRaySphericalVolume(void)
        : AbstractOSPRayStructure()
        , getDataSlot("getdata", "Connects to the data source")
        , getTFSlot("gettransferfunction", "Connects to a color transfer function module")
        , clippingBoxLower("ClippingBox::Left", "Left corner of the clipping Box")
        , clippingBoxUpper("ClippingBox::Right", "Right corner of the clipping Box")
        , clippingBoxActive("ClippingBox::Active", "Activates the clipping Box")
        , repType(
              "Representation", "Activates one of the three different volume representations: Volume, Isosurfae, Slice")
        , IsoValue("Isosurface::Isovalue", "Sets the isovalue of the isosurface")
        , showBoundingBox("showBoundingBox", "Bounding box of the volume data set")
        , volumeDataStringSlot("volumeData", "Set name for volume data from adios file") {

    core::param::EnumParam* rt = new core::param::EnumParam(VOLUMEREP);
    rt->SetTypePair(VOLUMEREP, "Volume");
    rt->SetTypePair(ISOSURFACE, "Isosurface");
    repType << rt;
    repType.SetUpdateCallback(&OSPRaySphericalVolume::paramChanged);
    MakeSlotAvailable(&repType);

    clippingBoxActive << new core::param::BoolParam(false);
    clippingBoxLower << new core::param::Vector3fParam(-5.0f, -5.0f, -5.0f);
    clippingBoxUpper << new core::param::Vector3fParam(0.0f, 5.0f, 5.0f);
    clippingBoxActive.SetUpdateCallback(&OSPRaySphericalVolume::paramChanged);
    clippingBoxLower.SetUpdateCallback(&OSPRaySphericalVolume::paramChanged);
    clippingBoxUpper.SetUpdateCallback(&OSPRaySphericalVolume::paramChanged);
    MakeSlotAvailable(&clippingBoxActive);
    MakeSlotAvailable(&clippingBoxLower);
    MakeSlotAvailable(&clippingBoxUpper);

    IsoValue << new core::param::FloatParam(0.1f);
    IsoValue.SetUpdateCallback(&OSPRaySphericalVolume::paramChanged);
    MakeSlotAvailable(&IsoValue);

    getDataSlot.SetCompatibleCall<adios::CallADIOSDataDescription>();
    MakeSlotAvailable(&getDataSlot);

    getTFSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    MakeSlotAvailable(&getTFSlot);

    showBoundingBox << new core::param::StringParam("");
    showBoundingBox.Parameter()->SetGUIReadOnly(true);
    MakeSlotAvailable(&showBoundingBox);

    volumeDataStringSlot << new core::param::FlexEnumParam("undef");
    volumeDataStringSlot.SetUpdateCallback(&OSPRaySphericalVolume::paramChanged);
    MakeSlotAvailable(&volumeDataStringSlot);


    // this->SetSlotUnavailable(&this->getMaterialSlot);
}


bool OSPRaySphericalVolume::readData(core::Call& call) {

    // fill material container
    this->processMaterial();

    // get transformation parameter
    //this->processTransformation();

    // fill clipping plane container
    this->processClippingPlane();

    // read Data, calculate  shape parameters, fill data vectors
    auto os = dynamic_cast<CallOSPRayStructure*>(&call);
    auto cd = this->getDataSlot.CallAs<adios::CallADIOSData>();
    auto const cgtf = this->getTFSlot.CallAs<core::view::CallGetTransferFunction>();

    if (cd == nullptr)
        return false;
    if (cgtf == nullptr) {
        core::utility::log::Log::DefaultLog.WriteError("[OSPRaySphericalVolume] No transferfunction connected.");
        return false;
    }

    uint32_t requested_frame = std::floorf(os->getTime());
    if ((requested_frame >= cd->getFrameCount() && cd->getFrameCount() > 0)) {
        requested_frame = cd->getFrameCount() - 1;
    }
    cd->setFrameIDtoLoad(requested_frame);

    // get meta data
    if (!(*cd)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[OSPRaySphericalVolume] Error during GetHeader");
        return false;
    }

    auto availVars = cd->getAvailableVars();
    for (auto var : availVars) {
        volumeDataStringSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
    }

    // do inqures
    const std::string volDataStr =
        std::string(this->volumeDataStringSlot.Param<core::param::FlexEnumParam>()->ValueString());
    if (volDataStr != "undef") {
        if (!cd->inquireVar(volDataStr)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[OSPRaySphericalVolume] variable \"%s\" doe not exist.", volDataStr.c_str());
            return false;
        }
    }
    const std::string bboxStr = "global_box";
    if (!cd->inquireVar(bboxStr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[OSPRaySphericalVolume] variable \"%s\" does not exist.", bboxStr.c_str());
    }
    const std::string gridOrigStr = "vol_grid_origin";
    if (!cd->inquireVar(gridOrigStr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[OSPRaySphericalVolume] variable \"%s\" does not exist.", gridOrigStr.c_str());
    }
    const std::string gridSpacingStr = "vol_grid_spacing";
    if (!cd->inquireVar(gridSpacingStr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[OSPRaySphericalVolume] variable \"%s\" does not exist.", gridSpacingStr.c_str());
    }
    const std::string gridResStr = "vol_grid_resolution";
    if (!cd->inquireVar(gridResStr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[OSPRaySphericalVolume] variable \"%s\" does not exist.", gridSpacingStr.c_str());
    }

    if (!(*cd)(0)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[OSPRaySphericalVolume] Error during GetData");
        return false;
    }

    // do the callback to set the dirty flag
    if (!(*cgtf)(0))
        return false;

    auto tf_dirty = cgtf->IsDirty();
    if (datahash != cd->getDataHash() || this->time != os->getTime() || _trigger_recalc || tf_dirty) {
        datahash = cd->getDataHash();
        time = os->getTime();
        structureContainer.dataChanged = true;
        _trigger_recalc = false;
    } else {
        return true;
    }

    auto bbox = cd->getData(bboxStr)
                    ->GetAsFloat();
    auto data = cd->getData(volDataStr)
                    ->GetAsFloat();
    auto gridOrigin = cd->getData(gridOrigStr)->GetAsFloat();
    auto gridSpacing = cd->getData(gridSpacingStr)->GetAsFloat();
    auto gridRes = cd->getData(gridResStr)->GetAsUInt32();

    this->extendContainer.boundingBox = std::make_shared<core::BoundingBoxes_2>();
    this->extendContainer.boundingBox->SetBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);

    std::stringstream bbox_stream;
    bbox_stream << "LEFT: " << extendContainer.boundingBox->BoundingBox().Left() << ";" << std::endl;
    bbox_stream << "BOTTOM: " << extendContainer.boundingBox->BoundingBox().Bottom() << ";" << std::endl;
    bbox_stream << "BACK: " << extendContainer.boundingBox->BoundingBox().Back() << ";" << std::endl;
    bbox_stream << "RIGHT: " << extendContainer.boundingBox->BoundingBox().Right() << ";" << std::endl;
    bbox_stream << "TOP: " << extendContainer.boundingBox->BoundingBox().Top() << ";" << std::endl;
    bbox_stream << "FRONT: " << extendContainer.boundingBox->BoundingBox().Front();
    this->showBoundingBox.Param<core::param::StringParam>()->SetValue(bbox_stream.str().c_str());
    this->extendContainer.timeFramesCount = cd->getFrameCount();
    this->extendContainer.isValid = true;



    unsigned int const voxelCount = gridRes[0] * gridRes[1] * gridRes[2];
    assert(data.size() == voxelCount);

    std::array<uint32_t, 3> const dimensions = {gridRes[0], gridRes[1], gridRes[2]};

    voxelDataType const voxelType = voxelDataType::FLOAT;

    // resort data
    _resorted_data.reserve(dimensions[0] * dimensions[1]*dimensions[2]);
    for (int phi = 0; phi < dimensions[2]; ++phi) {
        for (int theta = 0; theta < dimensions[1]; ++theta) {
            for (int r = 0; r < dimensions[0]; ++r) {
                _resorted_data.emplace_back(data[r + dimensions[0] * (theta + dimensions[1] * phi)]);
            }
        }
    }

    // get color transfer function
    std::vector<float> rgb;
    std::vector<float> a;
    std::array<float, 2> const minmax = {*std::min_element(data.begin(), data.end()), *std::max_element(data.begin(),data.end())};
    cgtf->SetRange(minmax);
    if ((*cgtf)(0)) {
        if (cgtf->TFTextureFormat() == core::view::CallGetTransferFunction::TextureFormat::TEXTURE_FORMAT_RGBA) {
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
            auto const texSize = cgtf->TextureSize();
            rgb.resize(3 * texSize);
            a.resize(texSize);
            auto const texture = cgtf->GetTextureData();

            for (unsigned int i = 0; i < texSize; ++i) {
                rgb[i * 3 + 0] = texture[i * 4 + 0];
                rgb[i * 3 + 1] = texture[i * 4 + 1];
                rgb[i * 3 + 2] = texture[i * 4 + 2];
                a[i] = i / (texSize - 1.0f);
            }
            core::utility::log::Log::DefaultLog.WriteWarn(
                "OSPRaySphericalVolume: No alpha channel in transfer function "
                "connected to module. Adding alpha ramp to RGB colors.\n");
        }
    } else {
        core::utility::log::Log::DefaultLog.WriteError(
            "OSPRaySphericalVolume: No transfer function connected to module");
        return false;
    }
    cgtf->ResetDirty();

    // Write stuff into the structureContainer

    this->structureContainer.type = structureTypeEnum::VOLUME;
    this->structureContainer.volumeType = volumeTypeEnum::SPHERICALVOLUME;
    volumeStructure svs;

    svs.volRepType = (volumeRepresentationType)this->repType.Param<core::param::EnumParam>()->Value();
    svs.voxels = _resorted_data.data();
    svs.gridOrigin = { gridOrigin[0], gridOrigin[1], gridOrigin[2]};
    svs.gridSpacing = { gridSpacing[0], gridSpacing[1], gridSpacing[2]};
    svs.dimensions = dimensions;
    svs.voxelCount = voxelCount;
    svs.valueRange = minmax;
    svs.tfRGB = std::make_shared<std::vector<float>>(std::move(rgb));
    svs.tfA = std::make_shared<std::vector<float>>(std::move(a));
    svs.voxelDType = voxelType;

    svs.clippingBoxActive = this->clippingBoxActive.Param<core::param::BoolParam>()->Value();
    std::array<float, 3> cbl = {this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetX(),
        this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetY(),
        this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().GetZ()};
    svs.clippingBoxLower = cbl;
    std::array<float, 3> cbu = {this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetX(),
        this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetY(),
        this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().GetZ()};
    svs.clippingBoxUpper = cbu;

    svs.isoValue = this->IsoValue.Param<core::param::FloatParam>()->Value();

    this->structureContainer.structure = svs;

    return true;
}


OSPRaySphericalVolume::~OSPRaySphericalVolume() {
    this->Release();
}

bool OSPRaySphericalVolume::create() {
    return true;
}

void OSPRaySphericalVolume::release() {}


bool OSPRaySphericalVolume::getExtends(core::Call& call) {
    auto os = dynamic_cast<CallOSPRayStructure*>(&call);
    auto cd = this->getDataSlot.CallAs<adios::CallADIOSData>();

    if (cd == nullptr)
        return false;

    if (!this->readData(call))
        return false;

    return true;
}

bool OSPRaySphericalVolume::paramChanged(core::param::ParamSlot& p) {

    _trigger_recalc = true;
    return true;
}
