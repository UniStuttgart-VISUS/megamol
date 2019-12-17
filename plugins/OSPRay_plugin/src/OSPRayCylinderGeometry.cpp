/*
 * OSPRayCylinderGeometry.cpp
 *
 * Copyright (C) 2019 by MegaMol Team. Alle Rechte vorbehalten.
 */
// Make crappy clang-format f*** off:
// clang-format off

#include "stdafx.h"

#include <ospray.h>

#include "mmcore/Call.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"

#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "OSPRay_plugin/CallOSPRayAPIObject.h"

#include "vislib/sys/Log.h"

#include "OSPRayCylinderGeometry.h"
#include "osputils.h"



/*
 * megamol::ospray::OSPRayCylinderGeometry::OSPRayCylinderGeometry
 */
megamol::ospray::OSPRayCylinderGeometry::OSPRayCylinderGeometry(void)
        : frameID((std::numeric_limits<unsigned int>::max)()),
        hashInput(0),
        hashState(0),
        paramRadius("Radius", "Specifies radius if no per-cylinder data is given."),
        paramScale("Scale", "Scales the length of the cylinder."),
        slotGetData("GetData", "Connects the data source."),
        slotInstantiate("Instantiate", "Allows OSPRay to instantiate the geometry.") {
    using namespace megamol::core::moldyn;
    using namespace megamol::core::param;
    using namespace megamol::core::view;

    /* Configure parameters. */
    this->paramRadius << new FloatParam(0.01f, 0.000001f);
    this->MakeSlotAvailable(&this->paramRadius);

    this->paramScale << new FloatParam(1.0f);
    this->MakeSlotAvailable(&this->paramScale);

    /* Configure slots. */
    this->slotGetData.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->slotGetData);

    this->slotInstantiate.SetCallback(CallOSPRayAPIObject::ClassName(),
        CallOSPRayAPIObject::FunctionName(0),
        &OSPRayCylinderGeometry::onGetData);
    this->slotInstantiate.SetCallback(CallOSPRayAPIObject::ClassName(),
        CallOSPRayAPIObject::FunctionName(1),
        &OSPRayCylinderGeometry::onGetDirty);
    this->slotInstantiate.SetCallback(CallOSPRayAPIObject::ClassName(),
        CallOSPRayAPIObject::FunctionName(2),
        &OSPRayCylinderGeometry::onGetExtents);
    this->MakeSlotAvailable(&this->slotInstantiate);
}


/*
 * megamol::ospray::OSPRayCylinderGeometry::~OSPRayCylinderGeometry
 */
megamol::ospray::OSPRayCylinderGeometry::~OSPRayCylinderGeometry(void) {
    this->Release();
}


/*
 * megamol::ospray::OSPRayCylinderGeometry::checkState
 */
bool megamol::ospray::OSPRayCylinderGeometry::checkState(
        core::param::ParamSlot& param, const bool reset) {
    auto retval = param.IsDirty();

    if (retval && reset) {
        param.ResetDirty();
    }

    return retval;
}


/*
 * megamol::ospray::OSPRayCylinderGeometry::getSizes
 */
std::tuple<std::size_t,std::size_t>
megamol::ospray::OSPRayCylinderGeometry::getSizes(
        const ParticleType& particles) {
    using vislib::sys::Log;
    std::size_t vertSize = 0;
    std::size_t dirSize = 0;

    switch (particles.GetDirDataType()) {
        case ParticleType::DirDataType::DIRDATA_FLOAT_XYZ:
            dirSize = 3 * sizeof(float);
            break;

        default:
            Log::DefaultLog.WriteWarn(_T("Input direction type %d is not ")
                _T("supported for OSPRay cylinders."),
                particles.GetDirDataType());
            ASSERT(dirSize == 0);
            break;
    }

    switch (particles.GetVertexDataType()) {
        case ParticleType::VertexDataType::VERTDATA_FLOAT_XYZ:
            vertSize = 3 * sizeof(float);
            break;

        case ParticleType::VertexDataType::VERTDATA_FLOAT_XYZR:
            vertSize = 4 * sizeof(float);
            break;

        default:
            Log::DefaultLog.WriteError(_T("Unsupported vertex type %d for ")
                _T("rendering OSPRay cylinders."),
                particles.GetVertexDataType());
            ASSERT(dirSize == 0);
            break;
    }

    return std::make_tuple(vertSize, dirSize);
}


/*
 * megamol::ospray::OSPRayCylinderGeometry::checkState
 */
bool megamol::ospray::OSPRayCylinderGeometry::checkState(const bool reset) {
    bool retval = false;

    retval |= OSPRayCylinderGeometry::checkState(this->paramRadius, reset);
    retval |= OSPRayCylinderGeometry::checkState(this->paramScale, reset);

    return retval;
}


/*
 * megamol::ospray::OSPRayCylinderGeometry::create
 */
bool megamol::ospray::OSPRayCylinderGeometry::create(void) {
    return true;
}


/*
 * megamol::ospray::OSPRayCylinderGeometry::getData
 */
bool megamol::ospray::OSPRayCylinderGeometry::getData(core::Call& call) {
    using namespace megamol::core;
    using vislib::sys::Log;

    auto gd = this->slotGetData.CallAs<moldyn::MultiParticleDataCall>();
    if (gd == nullptr) {
        Log::DefaultLog.WriteError(_T("No source was connected to GetData or ")
            _T("the source is not a MultiParticleDataCall."), nullptr);
        return false;
    }

    auto os = dynamic_cast<CallOSPRayStructure *>(std::addressof(call));
    if (os == nullptr) {
        Log::DefaultLog.WriteError(_T("The call to getExtents is not a ")
            _T("CallOSPRayStructure."), nullptr);
        return false;
    }

    gd->SetFrameID(os->getTime(), true);
    (*gd)(1);

    //this->extendContainer.boundingBox = std::make_shared<BoundingBoxes>(cd->AccessBoundingBoxes());
    //this->extendContainer.timeFramesCount = cd->FrameCount();
    //this->extendContainer.isValid = true;

    return true;
}


/*
 * megamol::ospray::OSPRayCylinderGeometry::onGetData
 */
bool megamol::ospray::OSPRayCylinderGeometry::onGetData(
        megamol::core::Call& call) {
    using namespace megamol::core;
    using namespace megamol::core::param;
    using vislib::sys::Log;

    auto gd = this->slotGetData.CallAs<moldyn::MultiParticleDataCall>();
    if (gd == nullptr) {
        Log::DefaultLog.WriteError(_T("No source was connected to GetData or ")
            _T("the source is not a MultiParticleDataCall."), nullptr);
        return false;
    }

    auto os = dynamic_cast<CallOSPRayAPIObject *>(std::addressof(call));
    if (os == nullptr) {
        Log::DefaultLog.WriteError(_T("The call to Instantiate is not a ")
            _T("CallOSPRayStructure."), nullptr);
        return false;
    }

    if (!(*gd)(1)) {
        Log::DefaultLog.WriteError(_T("Failed to get extents from source."),
            nullptr);
        return false;
    }

    const auto minFrameCount = gd->FrameCount();
    if (minFrameCount == 0) {
        Log::DefaultLog.WriteError(_T("The data source has no frames."),
            nullptr);
        return false;
    }

    auto frameTime = 0;
    if (os->FrameID() >= minFrameCount) {
        gd->SetFrameID(minFrameCount - 1, true); // isTimeForced flag set to true
        frameTime = minFrameCount - 1;
    } else {
        gd->SetFrameID(os->FrameID(), true); // isTimeForced flag set to true
        frameTime = os->FrameID();
    }

    // Determine whether the state of the module itself has changed.
    const auto isStateChanged = (this->frameID != frameTime)
        || this->checkState(true);
    if (isStateChanged) {
        ++this->hashState;
    }

    // Determine whether the data have changed or abort.
    if ((this->hashInput != gd->DataHash()) || isStateChanged) {
        this->frameID = frameTime;
        this->hashInput = gd->DataHash();
    } else {
        // Nothing to do here ...
        // It's a disaster!
        //return true;
    }

    if (!(*gd)(0)) {
        Log::DefaultLog.WriteError(_T("Failed to get data from source."),
            nullptr);
        return false;
    }

    std::vector<OSPGeometry> geo;
    geo.reserve(gd->GetParticleListCount());

    this->data.clear();

    for (unsigned int i = 0; i <  gd->GetParticleListCount(); ++i) {
        auto& particles = gd->AccessParticles(i);
        const auto cntParticles = particles.GetCount();
        const auto inputSizes = OSPRayCylinderGeometry::getSizes(particles);
        const auto scale = this->paramScale.Param<FloatParam>()->Value();

        // Check the data.
        if ((std::get<0>(inputSizes) == 0) || (std::get<1>(inputSizes) == 0)) {
            continue;
        }

        // Fill the data cache.
        auto vertices = static_cast<const std::int8_t *>(
            particles.GetVertexData());
        ASSERT(vertices != nullptr);
        auto directions = static_cast<const std::int8_t *>(
            particles.GetDirData());
        ASSERT(directions != nullptr);

        const auto offset = this->data.size();
        const auto outputSize = std::get<0>(inputSizes)
            + std::get<1>(inputSizes);
        this->data.resize(this->data.size() + cntParticles * outputSize);

        for (std::decay<decltype(cntParticles)>::type j = 0; j < cntParticles;
                ++j) {
            auto dst = this->data.data() + offset + j * outputSize;
            auto v0 = reinterpret_cast<float *>(dst);
            ::memcpy(dst, vertices + j * particles.GetVertexDataStride(),
                std::get<0>(inputSizes));

            dst += std::get<0>(inputSizes);
            auto v1 = reinterpret_cast<float *>(dst);
            ::memcpy(dst, directions + j * particles.GetDirDataStride(),
                std::get<1>(inputSizes));

            for (std::size_t k = 0; k < 3; ++k) {
                v1[k] = v0[k] + scale * v1[k];
            }
        }

        geo.push_back(::ospNewGeometry("cylinders"));
        ASSERT(!geo.empty());
        ASSERT(geo.back() != nullptr);

        // Pass the data to OSPRay.
        {
            auto ospData = ::ospNewData(cntParticles * outputSize,
                OSPDataType::OSP_CHAR,
                this->data.data() + offset,
                OSPDataCreationFlags::OSP_DATA_SHARED_BUFFER);
            ::ospCommit(ospData);
            ::ospSetData(geo.back(), "cylinders", ospData);
        }

        // Set parameters of OSPRay cylinder geometry.
        ::ospSet1f(geo.back(), "radius",
            this->paramRadius.Param<FloatParam>()->Value());
        // "materialID" is not supported.
        ::ospSet1i(geo.back(), "bytes_per_cylinder",
            outputSize);
        ::ospSet1i(geo.back(), "offset_v0",
            0);
        ::ospSet1i(geo.back(), "offset_v1",
            std::get<0>(inputSizes));

        switch (particles.GetVertexDataType()) {
            case ParticleType::VertexDataType::VERTDATA_FLOAT_XYZR:
                ::ospSet1i(geo.back(), "offset_radius",
                    3 * sizeof(float));
                break;

            default:
                ::ospSet1i(geo.back(), "offset_radius", -1);
                break;
        }

        // "offset_materialID" is not supported.
        // "offset_colorID" is not supported.
        ::ospCommit(geo.back());
    } /* end for (unsigned int i = 0; i <  gd->GetParticleListCount(); ++i) */

    {
        std::vector<void *> tmp;
        tmp.reserve(geo.size());
        std::transform(geo.begin(), geo.end(), std::back_inserter(tmp),
            [](OSPGeometry& g) { return static_cast<void *>(g); });
        os->setStructureType(GEOMETRY);
        os->setAPIObjects(std::move(tmp));
    }

    return true;
}


/*
 * megamol::ospray::OSPRayCylinderGeometry::onGetDirty
 */
bool megamol::ospray::OSPRayCylinderGeometry::onGetDirty(
        megamol::core::Call& call) {
    using namespace megamol::core;
    using vislib::sys::Log;

    auto gd = this->slotGetData.CallAs<moldyn::MultiParticleDataCall>();
    if (gd == nullptr) {
        Log::DefaultLog.WriteError(_T("No source was connected to GetData or ")
            _T("the source is not a MultiParticleDataCall."), nullptr);
        return false;
    }

    auto os = dynamic_cast<CallOSPRayAPIObject *>(std::addressof(call));
    if (os == nullptr) {
        Log::DefaultLog.WriteError(_T("The call to Instantiate is not a ")
            _T("CallOSPRayStructure."), nullptr);
        return false;
    }

    if (this->checkState(false)) {
        // TODO: This is as in OSPRayPDKGeometry, but seems dangerous to me ...
        // I have done it this way, because I do not know how the rest of the
        // code is using it.
        os->setDirty();
    }

    if (gd->DataHash() != this->hashInput) {
        // TODO: This seems dangerous as well as it does not cover state changes
        // in this module ...
        os->SetDataHash(this->getHash());
    }

    return true;
}


/*
 * megamol::ospray::OSPRayCylinderGeometry::onGetExtents
 */
bool megamol::ospray::OSPRayCylinderGeometry::onGetExtents(
        megamol::core::Call& call) {
    using namespace megamol::core;
    using vislib::sys::Log;

    auto gd = this->slotGetData.CallAs<moldyn::MultiParticleDataCall>();
    if (gd == nullptr) {
        Log::DefaultLog.WriteError(_T("No source was connected to GetData or ")
            _T("the source is not a MultiParticleDataCall."), nullptr);
        return false;
    }

    auto os = dynamic_cast<CallOSPRayAPIObject *>(std::addressof(call));
    if (os == nullptr) {
        Log::DefaultLog.WriteError(_T("The call to Instantiate is not a ")
            _T("CallOSPRayStructure."), nullptr);
        return false;
    }

    gd->SetFrameID(os->FrameID(), true);

    if (!(*gd)(1)) {
        Log::DefaultLog.WriteWarn(_T("The call to retrieve the source extents ")
            _T("failed. This must be ignored for the OSPRay stuff ..."),
            nullptr);
    }

    os->SetExtent(gd->FrameCount(), gd->AccessBoundingBoxes());

    return true;
}


/*
 * megamol::ospray::OSPRayCylinderGeometry::release
 */
void megamol::ospray::OSPRayCylinderGeometry::release(void) { }
