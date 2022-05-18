/*
 * OSPRayAOVSphereGeometry.cpp
 * Copyright (C) 2009-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "OSPRayAOVSphereGeometry.h"
#include "geometry_calls//MultiParticleDataCall.h"
#include "geometry_calls/VolumetricDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmospray/CallOSPRayAPIObject.h"
#include "ospray/ospray_cpp.h"
#include "ospray/ospray_cpp/ext/rkcommon.h"
#include "stdafx.h"


using namespace megamol::ospray;


typedef float (*floatFromArrayFunc)(const megamol::geocalls::MultiParticleDataCall::Particles& p, size_t index);
typedef unsigned char (*byteFromArrayFunc)(const megamol::geocalls::MultiParticleDataCall::Particles& p, size_t index);


OSPRayAOVSphereGeometry::OSPRayAOVSphereGeometry(void)
        : samplingRateSlot("samplingrate", "Set the samplingrate for the ao volume")
        , aoThresholdSlot(
              "aoThreshold", "Set the threshold for the ao vol sampling above which a sample is assumed to occlude")
        , aoRayOffsetFactorSlot("aoRayOffsetFactor", "Set the factor for AO ray offset, to avoid self intersection")
        , getDataSlot("getdata", "Connects to the data source")
        , getVolSlot("getVol", "Connects to the density volume provider")
        , deployStructureSlot("deployStructureSlot", "Connects to an OSPRayAPIStructure") {

    this->getDataSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getVolSlot.SetCompatibleCall<geocalls::VolumetricDataCallDescription>();
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
    auto cd = this->getDataSlot.CallAs<geocalls::MultiParticleDataCall>();
    auto vd = this->getVolSlot.CallAs<geocalls::VolumetricDataCall>();
    if (vd == nullptr)
        return false;

    if (cd == nullptr)
        return false;

    auto particleExtentsOK = (*cd)(1);
    auto volExtentsOK = (*vd)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS);

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

    auto volMetaOK = (*vd)(geocalls::VolumetricDataCall::IDX_GET_METADATA);
    auto volDataOK = (*vd)(geocalls::VolumetricDataCall::IDX_GET_DATA);


    if (cd->GetParticleListCount() == 0)
        recompute = false;

    static bool isInitAOV = false;

    // START OSPRAY STUFF

    if (!isInitAOV) {
        auto error = ospLoadModule("aovspheres");
        if (error != OSP_NO_ERROR) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to load OSPRay module: AOVSpheres. Error occured in %s:%d", __FILE__, __LINE__);
            return false;
        } else {
            isInitAOV = true;
        }
    }

    ::ospray::cpp::Volume aovol = NULL;
    ::ospray::cpp::VolumetricModel aovol_model;
    std::vector<::ospray::cpp::Geometry> geo;
    if (!volDataOK) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "OSPRayAOVSphereGeometry: connected module supplies no volume data, ignore subsequent errors\n");
    }
    if (!volMetaOK) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "OSPRayAOVSphereGeometry: connected module supplies no volume metadata, ignore subsequent errors\n");
    }
    if (!volExtentsOK) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "OSPRayAOVSphereGeometry: connected module supplies no volume extents, ignore subsequent errors\n");
    }
    if (!particleDataOK) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "OSPRayAOVSphereGeometry: connected module supplies no particle data, ignore subsequent errors\n");
    }
    if (!particleExtentsOK) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "OSPRayAOVSphereGeometry: connected module supplies no particle extents, ignore subsequent errors\n");
    }

    if (volDataOK && volMetaOK && volExtentsOK && particleDataOK && particleExtentsOK && recompute) {
        for (unsigned int plist = 0; plist < cd->GetParticleListCount(); ++plist) {

            geocalls::MultiParticleDataCall::Particles& parts = cd->AccessParticles(plist);

            if (parts.GetVertexDataType() == geocalls::SimpleSphericalParticles::VERTDATA_NONE)
                continue;

            unsigned int const partCount = parts.GetCount();
            float const globalRadius = parts.GetGlobalRadius();

            size_t colorLength;
            size_t vertexLength;

            // Vertex data type check
            if (parts.GetVertexDataType() == geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) {
                vertexLength = 3;
            } else if (parts.GetVertexDataType() == geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
                vertexLength = 4;
            }

            // Color data type check
            if (parts.GetColourDataType() == geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA) {
                colorLength = 4;
            } else if (parts.GetColourDataType() == geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
                colorLength = 1;
            } else if (parts.GetColourDataType() == geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB) {
                colorLength = 3;
            } else if (parts.GetColourDataType() == geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA) {
                colorLength = 4;
            } else if (parts.GetColourDataType() == geocalls::MultiParticleDataCall::Particles::COLDATA_NONE) {
                colorLength = 0;
            }

            unsigned long long vertStride = parts.GetVertexDataStride();
            if (vertStride == 0) {
                vertStride = geocalls::MultiParticleDataCall::Particles::VertexDataSize[parts.GetVertexDataType()];
            }

            if (parts.GetVertexDataType() == geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE &&
                parts.GetColourDataType() != geocalls::MultiParticleDataCall::Particles::COLDATA_NONE) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("Only color data is not allowed.");
            }

            // get the volume stuff
            auto const volSDT = vd->GetScalarType(); //< unfortunately only float is part of the intersection
            if (volSDT != geocalls::VolumetricDataCall::ScalarType::FLOATING_POINT) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "OSPRayAOVSphereGeometry: Only float is supported as AOVol data type\n");
                continue;
            }
            auto const volGT = vd->GetGridType();
            if (volGT != geocalls::VolumetricDataCall::GridType::CARTESIAN &&
                volGT != geocalls::VolumetricDataCall::GridType::RECTILINEAR) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "OSPRayAOVSphereGeometry: Currently only grids are supported as AOVol grid type\n");
                continue;
            }
            auto const volCom = vd->GetComponents();
            if (volCom != 1) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "OSPRayAOVSphereGeometry: Only one component per cell is allowed as AOVol\n");
                continue;
            }
            auto const metadata = vd->GetMetadata();
            if (metadata->MinValues == nullptr || metadata->MaxValues == nullptr) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "OSPRayAOVSphereGeometry: AOVol requires a specified value range\n");
                recompute = false;
            }
            float const minV = metadata->MinValues[0];
            float const maxV = metadata->MaxValues[0];
            this->valuerange = {minV, maxV};
            this->gridorigin = {metadata->Origin[0], metadata->Origin[1], metadata->Origin[2]};
            this->gridspacing = {metadata->SliceDists[0][0], metadata->SliceDists[1][0], metadata->SliceDists[2][0]};
            this->dimensions = {static_cast<int>(metadata->Resolution[0]), static_cast<int>(metadata->Resolution[1]),
                static_cast<int>(metadata->Resolution[2])}; //< TODO HAZARD explizit narrowing

            float cellVol = metadata->SliceDists[0][0] * metadata->SliceDists[1][0] * metadata->SliceDists[2][0];
            const float valRange = maxV - minV;


            const auto numCreateGeo = parts.GetCount() * vertStride / ispcLimit + 1;

            // geo.resize(geo.size() + numCreateGeo);
            for (unsigned int i = 0; i < numCreateGeo; i++) {

                if (parts.GetCount() == 0)
                    continue;
                geo.emplace_back(::ospray::cpp::Geometry("aovspheres_geometry"));
                unsigned long long floatsToRead = parts.GetCount() * vertStride / (numCreateGeo * sizeof(float));
                floatsToRead -= floatsToRead % (vertStride / sizeof(float));

                auto vertexData =
                    ::ospray::cpp::SharedData(&static_cast<const float*>(parts.GetVertexData())[i * floatsToRead],
                        OSP_FLOAT, floatsToRead / 3, vertStride);

                vertexData.commit();
                geo.back().setParam("spheres", vertexData);
                geo.back().setParam("color", NULL);

                if (vertexLength > 3) {
                    geo.back().setParam("offset_radius", 3 * sizeof(float));
                } else {
                    geo.back().setParam("radius", globalRadius);
                }
                if (parts.GetColourDataType() ==
                        geocalls::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_RGB ||
                    parts.GetColourDataType() ==
                        geocalls::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_RGBA) {

                    geo.back().setParam("color_offset",
                        vertexLength * sizeof(float)); // TODO: This won't work if there are radii in the array
                    geo.back().setParam("color_stride", parts.GetColourDataStride());
                    geo.back().setParam("color", vertexData);
                    if (parts.GetColourDataType() ==
                        geocalls::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_RGB) {
                        geo.back().setParam("color_format", OSP_VEC3F);
                    } else {
                        geo.back().setParam("color_format", OSP_VEC4F);
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

                    aovol = ::ospray::cpp::Volume("shared_structured_volume");


                    // add data
                    rkcommon::math::vec3i dims = {this->dimensions[0], this->dimensions[1], this->dimensions[2]};
                    auto voxelData =
                        ::ospray::cpp::SharedData(vd->GetData(), OSP_FLOAT, dims, rkcommon::math::vec3i(0));
                    voxelData.commit();


                    aovol.setParam("data", voxelData);

                    rkcommon::math::vec3f gorigin = {this->gridorigin[0], this->gridorigin[1], this->gridorigin[2]};
                    aovol.setParam("gridOrigin", gorigin);
                    rkcommon::math::vec3f gspacing = {this->gridspacing[0], this->gridspacing[1], this->gridspacing[2]};
                    aovol.setParam("gridSpacing", gspacing);


                    auto tf = ::ospray::cpp::TransferFunction("piecewiseLinear");

                    std::vector<float> faketf = {
                        1.0f,
                        1.0f,
                        1.0f,
                        1.0f,
                        1.0f,
                        1.0f,
                    };
                    std::vector<float> fakeopa = {1.0f, 1.0f};

                    auto tf_rgb = ::ospray::cpp::CopiedData(faketf.data(), OSP_FLOAT, faketf.size() / 3);
                    auto tf_opa = ::ospray::cpp::CopiedData(fakeopa);
                    tf.setParam("color", tf_rgb);
                    tf.setParam("opacity", tf_opa);
                    rkcommon::math::vec2f valrange = {this->valuerange[0], this->valuerange[1]};
                    tf.setParam("valueRange", valrange);

                    tf.commit();

                    aovol.commit();

                    aovol_model = ::ospray::cpp::VolumetricModel(aovol);

                    aovol_model.setParam("transferFunction", tf);
                    aovol_model.commit();


                    //aovol, "voxelRange", this->valuerange.first, this->valuerange.second);
                    //ospSet1f(aovol, "samplingRate", this->samplingRateSlot.Param<core::param::FloatParam>()->Value());
                    // ospSet1b(aovol, "adaptiveSampling", false);


                    /*auto ptr = element.raw2.get();
                    ospSetRegion(aovol, ptr, osp::vec3i{0, 0, 0},
                        osp::vec3i{(*element.dimensions)[0], (*element.dimensions)[1], (*element.dimensions)[2]});*/

                    /*ospStructures.push_back(std::make_pair(aovol, structureTypeEnum::VOLUME));*/
                }

                assert(aovol);

                geo.back().setParam(
                    "aothreshold", valRange * this->aoThresholdSlot.Param<core::param::FloatParam>()->Value());
                geo.back().setParam("aoRayOffset",
                    maxGridSpacing * this->aoRayOffsetFactorSlot.Param<core::param::FloatParam>()->Value());
                geo.back().setParam("aovol", ::ospray::cpp::CopiedData(aovol_model));
                // ospCommit(geo);
            } // geometries
        }     // particle lists
    }

    std::vector<void*> geo_transfer(geo.size());
    for (auto i = 0; i < geo.size(); i++) {
        geo_transfer[i] = geo[i].handle();
    }
    os->setStructureType(GEOMETRY);
    os->setAPIObjects(std::move(geo_transfer));

    return true;
}

OSPRayAOVSphereGeometry::~OSPRayAOVSphereGeometry() {
    this->Release();
}


bool OSPRayAOVSphereGeometry::create() {
    return true;
}


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
    auto cd = this->getDataSlot.CallAs<geocalls::MultiParticleDataCall>();

    if (cd == nullptr)
        return false;
    cd->SetFrameID(os->FrameID());
    (*cd)(1);
    core::BoundingBoxes_2 box;
    box.SetBoundingBox(cd->AccessBoundingBoxes().ObjectSpaceBBox());
    os->SetExtent(cd->FrameCount(), box);

    return true;
}

bool megamol::ospray::OSPRayAOVSphereGeometry::getDirtyCallback(core::Call& call) {
    auto os = dynamic_cast<CallOSPRayAPIObject*>(&call);
    auto cd = this->getDataSlot.CallAs<geocalls::MultiParticleDataCall>();

    if (cd == nullptr)
        return false;
    if (this->InterfaceIsDirtyNoReset()) {
        os->setDirty();
    }
    if (this->datahash != cd->DataHash()) {
        os->SetDataHash(cd->DataHash());
    }
    return true;
}
