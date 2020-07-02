/*
 * OSPRayAOVSphereGeometry.cpp
 * Copyright (C) 2009-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "OSPRayAOVSphereGeometry.h"
#include "../../protein/src/Color.h"
#include "OSPRay_plugin/CallOSPRayAPIObject.h"
#include "mmcore/Call.h"
#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "ospray/ospray.h"
#include "vislib/sys/Log.h"


using namespace megamol::ospray;


typedef float (*floatFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);
typedef unsigned char (*byteFromArrayFunc)(
    const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);


OSPRayAOVSphereGeometry::OSPRayAOVSphereGeometry(void)
    : samplingRateSlot("samplingrate", "Set the samplingrate for the ao volume")
    , aoThresholdSlot(
          "aoThreshold", "Set the threshold for the ao vol sampling above which a sample is assumed to occlude")
    , aoRayOffsetFactorSlot("aoRayOffsetFactor", "Set the factor for AO ray offset, to avoid self intersection")
    , getDataSlot("getdata", "Connects to the data source")
    , getVolSlot("getVol", "Connects to the density volume provider")
    , deployStructureSlot("deployStructureSlot", "Connects to an OSPRayAPIStructure") {

    this->getDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getVolSlot.SetCompatibleCall<core::misc::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->getVolSlot);

    this->samplingRateSlot << new core::param::FloatParam(1.0f, 0.0f, std::numeric_limits<float>::max());
    this->MakeSlotAvailable(&this->samplingRateSlot);

    // this->aoThresholdSlot << new core::param::FloatParam(0.5f, 0.0f, 1.0f);
    this->aoThresholdSlot << new core::param::FloatParam(0.5f, 0.0f, std::numeric_limits<float>::max());
    this->MakeSlotAvailable(&this->aoThresholdSlot);

    this->aoRayOffsetFactorSlot << new core::param::FloatParam(1.0f, 0.0f);
    this->MakeSlotAvailable(&this->aoRayOffsetFactorSlot);


    this->deployStructureSlot.SetCallback(CallOSPRayAPIObject::ClassName(), CallOSPRayAPIObject::FunctionName(0),
        &OSPRayAOVSphereGeometry::getDataCallback);
    this->deployStructureSlot.SetCallback(CallOSPRayAPIObject::ClassName(), CallOSPRayAPIObject::FunctionName(1),
        &OSPRayAOVSphereGeometry::getExtendsCallback);
    this->deployStructureSlot.SetCallback(CallOSPRayAPIObject::ClassName(), CallOSPRayAPIObject::FunctionName(2),
        &OSPRayAOVSphereGeometry::getDirtyCallback);
    this->MakeSlotAvailable(&this->deployStructureSlot);
}


bool OSPRayAOVSphereGeometry::getDataCallback(megamol::core::Call& call) {

    // read Data, calculate  shape parameters, fill data vectors
    auto os = dynamic_cast<CallOSPRayAPIObject*>(&call);
    auto cd = this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();
    auto vd = this->getVolSlot.CallAs<core::misc::VolumetricDataCall>();
    if (vd == nullptr) return false;

    if (cd == nullptr) return false;

    auto particleExtentsOK = (*cd)(1);
    auto volExtentsOK = (*vd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS);

    auto const minFrameCount = std::min(cd->FrameCount(), vd->FrameCount());

    // if (minFrameCount == 0) return false;

    auto frameTime = 0;

    if (os->FrameID() >= minFrameCount) {
        cd->SetFrameID(minFrameCount - 1, true); // isTimeForced flag set to true
        vd->SetFrameID(minFrameCount - 1, true); // isTimeForced flag set to true
        frameTime = minFrameCount - 1;
    } else {
        cd->SetFrameID(os->FrameID(), true); // isTimeForced flag set to true
        vd->SetFrameID(os->FrameID(), true); // isTimeForced flag set to true
        frameTime = os->FrameID();
    }

    bool recompute = true;
    if (this->datahash != cd->DataHash() || this->time != frameTime || this->volDatahash != vd->DataHash() ||
        this->volFrameID != frameTime || this->InterfaceIsDirty()) {
        this->datahash = cd->DataHash();
        this->time = frameTime;
        this->volDatahash = vd->DataHash();
        this->volFrameID = frameTime;
    } else {
        recompute = false;
    }

    auto particleDataOK = (*cd)(0);

    auto volMetaOK = (*vd)(core::misc::VolumetricDataCall::IDX_GET_METADATA);
    auto volDataOK = (*vd)(core::misc::VolumetricDataCall::IDX_GET_DATA);


    if (cd->GetParticleListCount() == 0) recompute = false;

    static bool isInitAOV = false;

    // START OSPRAY STUFF

    if (!isInitAOV) {
        auto error = ospLoadModule("aovspheres");
        if (error != OSP_NO_ERROR) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Unable to load OSPRay module: AOVSpheres. Error occured in %s:%d", __FILE__, __LINE__);
            return false;
        } else {
            isInitAOV = true;
        }
    }

    OSPVolume aovol = NULL;
    std::vector<OSPGeometry> geo;
    if (!volDataOK) {
        vislib::sys::Log::DefaultLog.WriteError(
            "OSPRayAOVSphereGeometry: connected module supplies no volume data, ignore subsequent errors\n");
    }
    if (!volMetaOK) {
        vislib::sys::Log::DefaultLog.WriteError(
            "OSPRayAOVSphereGeometry: connected module supplies no volume metadata, ignore subsequent errors\n");
    }
    if (!volExtentsOK) {
        vislib::sys::Log::DefaultLog.WriteError(
            "OSPRayAOVSphereGeometry: connected module supplies no volume extents, ignore subsequent errors\n");
    }
    if (!particleDataOK) {
        vislib::sys::Log::DefaultLog.WriteError(
            "OSPRayAOVSphereGeometry: connected module supplies no particle data, ignore subsequent errors\n");
    }
    if (!particleExtentsOK) {
        vislib::sys::Log::DefaultLog.WriteError(
            "OSPRayAOVSphereGeometry: connected module supplies no particle extents, ignore subsequent errors\n");
    }

    if (volDataOK && volMetaOK && volExtentsOK && particleDataOK && particleExtentsOK && recompute) {
        for (unsigned int plist = 0; plist < cd->GetParticleListCount(); ++plist) {

            core::moldyn::MultiParticleDataCall::Particles& parts = cd->AccessParticles(plist);

            if (parts.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_NONE) continue;

            unsigned int const partCount = parts.GetCount();
            float const globalRadius = parts.GetGlobalRadius();

            size_t colorLength;
            size_t vertexLength;

            // Vertex data type check
            if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) {
                vertexLength = 3;
            } else if (parts.GetVertexDataType() ==
                       core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
                vertexLength = 4;
            }

            // Color data type check
            if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA) {
                colorLength = 4;
            } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
                colorLength = 1;
            } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB) {
                colorLength = 3;
            } else if (parts.GetColourDataType() ==
                       core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA) {
                colorLength = 4;
            } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE) {
                colorLength = 0;
            }

            int vertStride = parts.GetVertexDataStride();
            if (vertStride == 0) {
                vertStride = core::moldyn::MultiParticleDataCall::Particles::VertexDataSize[parts.GetVertexDataType()];
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
                continue;
            }
            auto const volGT = vd->GetGridType();
            if (volGT != core::misc::VolumetricDataCall::GridType::CARTESIAN &&
                volGT != core::misc::VolumetricDataCall::GridType::RECTILINEAR) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "OSPRayAOVSphereGeometry: Currently only grids are supported as AOVol grid type\n");
                continue;
            }
            auto const volCom = vd->GetComponents();
            if (volCom != 1) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "OSPRayAOVSphereGeometry: Only one component per cell is allowed as AOVol\n");
                continue;
            }
            auto const metadata = vd->GetMetadata();
            if (metadata->MinValues == nullptr || metadata->MaxValues == nullptr) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "OSPRayAOVSphereGeometry: AOVol requires a specified value range\n");
                recompute = false;
            }
            float const minV = metadata->MinValues[0];
            float const maxV = metadata->MaxValues[0];
            this->valuerange = std::make_pair(minV, maxV);
            this->gridorigin = {metadata->Origin[0], metadata->Origin[1], metadata->Origin[2]};
            this->gridspacing = {metadata->SliceDists[0][0], metadata->SliceDists[1][0], metadata->SliceDists[2][0]};
            this->dimensions = {static_cast<int>(metadata->Resolution[0]), static_cast<int>(metadata->Resolution[1]),
                static_cast<int>(metadata->Resolution[2])}; //< TODO HAZARD explizit narrowing

            float cellVol = metadata->SliceDists[0][0] * metadata->SliceDists[1][0] * metadata->SliceDists[2][0];
            const float valRange = maxV - minV;


            const auto numCreateGeo = parts.GetCount() * vertStride / ispcLimit + 1;

            // geo.resize(geo.size() + numCreateGeo);
            for (unsigned int i = 0; i < numCreateGeo; i++) {

                if (parts.GetCount() == 0) continue;
                geo.emplace_back(ospNewGeometry("aovspheres_geometry"));
                long long int floatsToRead = parts.GetCount() * vertStride / (numCreateGeo * sizeof(float));
                floatsToRead -= floatsToRead % (vertStride / sizeof(float));

                auto vertexData = ospNewData(floatsToRead / 3, OSP_FLOAT3,
                    &static_cast<const float*>(parts.GetVertexData())[i * floatsToRead], OSP_DATA_SHARED_BUFFER);

                ospCommit(vertexData);
                ospSet1i(geo.back(), "bytes_per_sphere", vertStride);
                ospSetData(geo.back(), "spheres", vertexData);
                ospSetData(geo.back(), "color", nullptr);

                if (vertexLength > 3) {
                    ospSet1f(geo.back(), "offset_radius", 3 * sizeof(float));
                } else {
                    ospSet1f(geo.back(), "radius", globalRadius);
                }
                if (parts.GetColourDataType() ==
                        core::moldyn::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_RGB ||
                    parts.GetColourDataType() ==
                        core::moldyn::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_RGBA) {

                    ospSet1i(geo.back(), "color_offset",
                        vertexLength * sizeof(float)); // TODO: This won't work if there are radii in the array
                    ospSet1i(geo.back(), "color_stride", parts.GetColourDataStride());
                    ospSetData(geo.back(), "color", vertexData);
                    if (parts.GetColourDataType() ==
                        core::moldyn::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_RGB) {
                        // ospSet1i(geo.back(), "color_components", 3);
                        ospSet1i(geo.back(), "color_format", OSP_FLOAT3);
                    } else {
                        // ospSet1i(geo.back(), "color_components", 4);
                        ospSet1i(geo.back(), "color_format", OSP_FLOAT4);
                    }
                }

                /*float fixedSpacing[3];
                for (auto x = 0; x < 3; ++x) {
                    fixedSpacing[x] = this->gridspacing.at(x) / (this->dimensions.at(x) - 1) + this->gridspacing.at(x);
                }
                float maxGridSpacing = std::max(fixedSpacing[0], std::max(fixedSpacing[1], fixedSpacing[2]));*/
                float maxGridSpacing =
                    std::max(this->gridspacing.at(0), std::max(this->gridspacing.at(1), this->gridspacing.at(2)));
                // aovol
                // auto const aovol = ospNewVolume("block_bricked_volume");
                if (aovol == NULL) {
                    aovol = ospNewVolume("shared_structured_volume");
                    ospSet2f(aovol, "voxelRange", this->valuerange.first, this->valuerange.second);
                    ospSet1f(aovol, "samplingRate", this->samplingRateSlot.Param<core::param::FloatParam>()->Value());
                    // ospSet1b(aovol, "adaptiveSampling", false);
                    ospSet3iv(aovol, "dimensions", this->dimensions.data());
                    ospSetString(
                        aovol, "voxelType", voxelDataTypeS[static_cast<uint8_t>(voxelDataType::FLOAT)].c_str());
                    ospSet3fv(aovol, "gridOrigin", this->gridorigin.data());

                    ospSet3fv(aovol, "gridSpacing", this->gridspacing.data());

                    OSPTransferFunction tf = ospNewTransferFunction("piecewise_linear");

                    std::vector<float> faketf = {
                        1.0f,
                        1.0f,
                        1.0f,
                        1.0f,
                        1.0f,
                        1.0f,
                    };
                    std::vector<float> fakeopa = {1.0f, 1.0f};

                    OSPData tf_rgb = ospNewData(2, OSP_FLOAT3, faketf.data());
                    OSPData tf_opa = ospNewData(2, OSP_FLOAT, fakeopa.data());
                    ospSetData(tf, "colors", tf_rgb);
                    ospSetData(tf, "opacities", tf_opa);
                    ospSet2f(tf, "valueRange", 0.0f, 1.0f);

                    ospCommit(tf);

                    ospSetObject(aovol, "transferFunction", tf);
                    ospRelease(tf);

                    // add data
                    auto voxelcount = this->dimensions[0] * this->dimensions[1] * this->dimensions[2];
                    auto voxels = ospNewData(voxelcount,
                        static_cast<OSPDataType>(voxelDataTypeOSP[static_cast<uint8_t>(voxelDataType::FLOAT)]),
                        vd->GetData(), OSP_DATA_SHARED_BUFFER);
                    ospCommit(voxels);
                    ospSetData(aovol, "voxelData", voxels);

                    /*auto ptr = element.raw2.get();
                    ospSetRegion(aovol, ptr, osp::vec3i{0, 0, 0},
                        osp::vec3i{(*element.dimensions)[0], (*element.dimensions)[1], (*element.dimensions)[2]});*/

                    ospCommit(aovol);

                    // ospStructures.push_back(std::make_pair(aovol, structureTypeEnum::VOLUME));
                }

                assert(aovol);

                ospSet1f(geo.back(), "aothreshold",
                    valRange * this->aoThresholdSlot.Param<core::param::FloatParam>()->Value());
                ospSet1f(geo.back(), "aoRayOffset",
                    maxGridSpacing * this->aoRayOffsetFactorSlot.Param<core::param::FloatParam>()->Value());
                ospSetObject(geo.back(), "aovol", aovol);
                // ospCommit(geo);
            } // geometries
        }     // particle lists
    }

    std::vector<void*> geo_transfer(geo.size());
    for (auto i = 0; i < geo.size(); i++) {
        geo_transfer[i] = reinterpret_cast<void*>(geo[i]);
    }
    os->setStructureType(GEOMETRY);
    os->setAPIObjects(std::move(geo_transfer));

    return true;
}

OSPRayAOVSphereGeometry::~OSPRayAOVSphereGeometry() { this->Release(); }


bool OSPRayAOVSphereGeometry::create() { return true; }


void OSPRayAOVSphereGeometry::release() {}


bool OSPRayAOVSphereGeometry::InterfaceIsDirty() {
    if (this->aoThresholdSlot.IsDirty() || this->samplingRateSlot.IsDirty() || this->aoRayOffsetFactorSlot.IsDirty()) {
        this->aoThresholdSlot.ResetDirty();
        this->samplingRateSlot.ResetDirty();
        this->aoRayOffsetFactorSlot.ResetDirty();
        return true;
    } else {
        return false;
    }
}


bool megamol::ospray::OSPRayAOVSphereGeometry::InterfaceIsDirtyNoReset() const {
    if (this->aoThresholdSlot.IsDirty() || this->samplingRateSlot.IsDirty() || this->aoRayOffsetFactorSlot.IsDirty()) {
        return true;
    } else {
        return false;
    }
}


bool OSPRayAOVSphereGeometry::getExtendsCallback(megamol::core::Call& call) {
    auto os = dynamic_cast<CallOSPRayAPIObject*>(&call);
    auto cd = this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();

    if (cd == nullptr) return false;
    cd->SetFrameID(os->FrameID());
    (*cd)(1);
    core::BoundingBoxes_2 box;
    box.SetBoundingBox(cd->AccessBoundingBoxes().ObjectSpaceBBox());
    os->SetExtent(cd->FrameCount(), box);

    return true;
}

bool megamol::ospray::OSPRayAOVSphereGeometry::getDirtyCallback(core::Call& call) {
    auto os = dynamic_cast<CallOSPRayAPIObject*>(&call);
    auto cd = this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();

    if (cd == nullptr) return false;
    if (this->InterfaceIsDirtyNoReset()) {
        os->setDirty();
    }
    if (this->datahash != cd->DataHash()) {
        os->SetDataHash(cd->DataHash());
    }
    return true;
}
