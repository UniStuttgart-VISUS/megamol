/*
 * OSPRayArrowGeometry.cpp
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

#include "OSPRayArrowGeometry.h"



/*
 * megamol::ospray::OSPRayArrowGeometry::OSPRayArrowGeometry
 */
megamol::ospray::OSPRayArrowGeometry::OSPRayArrowGeometry(void)
        : frameID((std::numeric_limits<unsigned int>::max)()),
        hashInput(0),
        hashState(0),
        paramBaseRadius("BaseRadius", "Specifies the base radius of the arrows' cylinder."),
        paramScale("Scale", "A scaling factor that is applied to the length of all arrows."),
        paramTipLength("TipLength", "The relative length of the arrows' tips."),
        paramTipRadius("TipRadius", "The relative radius of the arrows' tips."),
        slotGetData("GetData", "Connects the data source."),
        slotInstantiate("Instantiate", "Allows OSPRay to instantiate the geometry.") {
    using namespace megamol::core::moldyn;
    using namespace megamol::core::param;
    using namespace megamol::core::view;

    /* Configure parameters. */
    this->paramBaseRadius << new FloatParam(0.01f, 0.001f);
    this->MakeSlotAvailable(&this->paramBaseRadius);

    this->paramScale << new FloatParam(1.0f, 0.0f);
    this->MakeSlotAvailable(&this->paramScale);

    this->paramTipLength << new FloatParam(0.25f, 0.0f, 1.0f);
    this->MakeSlotAvailable(&this->paramTipLength);

    this->paramTipRadius << new FloatParam(1.1f, 1.0f);
    this->MakeSlotAvailable(&this->paramTipRadius);

    /* Configure slots. */
    this->slotGetData.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->slotGetData);

    this->slotInstantiate.SetCallback(CallOSPRayAPIObject::ClassName(),
        CallOSPRayAPIObject::FunctionName(0),
        &OSPRayArrowGeometry::onGetData);
    this->slotInstantiate.SetCallback(CallOSPRayAPIObject::ClassName(),
        CallOSPRayAPIObject::FunctionName(1),
        &OSPRayArrowGeometry::onGetDirty);
    this->slotInstantiate.SetCallback(CallOSPRayAPIObject::ClassName(),
        CallOSPRayAPIObject::FunctionName(2),
        &OSPRayArrowGeometry::onGetExtents);
    this->MakeSlotAvailable(&this->slotInstantiate);
}


/*
 * megamol::ospray::OSPRayArrowGeometry::~OSPRayArrowGeometry
 */
megamol::ospray::OSPRayArrowGeometry::~OSPRayArrowGeometry(void) {
    this->Release();
}


/*
 * megamol::ospray::OSPRayArrowGeometry::checkParticles
 */
bool megamol::ospray::OSPRayArrowGeometry::checkParticles(
        const ParticleType& particles) {
    using namespace megamol::core::moldyn;
    using vislib::sys::Log;

    if (OSPRayArrowGeometry::toOspray(particles.GetVertexDataType())
            == OSPDataType::OSP_UNKNOWN) {
        return false;
    }

    if (OSPRayArrowGeometry::toOspray(particles.GetDirDataType())
            == OSPDataType::OSP_UNKNOWN) {
        return false;
    }

    // TODO: integrate colour.

    // Check whether our assumptions on the vertex layout hold.
    {
        auto vertices = static_cast<const std::int8_t *>(
            particles.GetVertexData());
        auto vertexStride = particles.GetVertexDataStride();

        auto directions = static_cast<const std::int8_t *>(
            particles.GetDirData());
        auto directionStride = particles.GetDirDataStride();

        if ((vertexStride != directionStride)
                || (std::abs(directions - vertices) > vertexStride)) {
            Log::DefaultLog.WriteWarn(_T("All per-arrow data need to be ")
                _T("laid out in interleaved rows for OSPRay arrows."), nullptr);
            return false;
        }
    }

    /* No problem found at this point. */
    return true;
}


/*
 * megamol::ospray::OSPRayArrowGeometry::checkState
 */
bool megamol::ospray::OSPRayArrowGeometry::checkState(
        core::param::ParamSlot& param, const bool reset) {
    auto retval = param.IsDirty();

    if (retval && reset) {
        param.ResetDirty();
    }

    return retval;
}


/*
 * megamol::ospray::OSPRayArrowGeometry::toOspray
 */
constexpr OSPDataType megamol::ospray::OSPRayArrowGeometry::toOspray(
        const ParticleType::DirDataType type) {
    using vislib::sys::Log;

    switch (type) {
        case ParticleType::DirDataType::DIRDATA_FLOAT_XYZ:
            return OSPDataType::OSP_FLOAT3;

        default:
            Log::DefaultLog.WriteWarn(_T("Unsupported directional data type ")
                _T("%d for OSPRay arrows."), type);
            return OSPDataType::OSP_UNKNOWN;
    }
}


/*
 * megamol::ospray::OSPRayArrowGeometry::toOspray
 */
constexpr OSPDataType megamol::ospray::OSPRayArrowGeometry::toOspray(
        const ParticleType::VertexDataType type) {
    using vislib::sys::Log;

    switch (type) {
        case ParticleType::VertexDataType::VERTDATA_FLOAT_XYZ:
            return OSPDataType::OSP_FLOAT3;

        case ParticleType::VertexDataType::VERTDATA_FLOAT_XYZR:
            return OSPDataType::OSP_FLOAT4;

        default:
            Log::DefaultLog.WriteWarn(_T("Unsupported data vertex type ")
                _T("%d for OSPRay arrows."), type);
            return OSPDataType::OSP_UNKNOWN;
    }
}


/*
 * megamol::ospray::OSPRayArrowGeometry::checkState
 */
bool megamol::ospray::OSPRayArrowGeometry::checkState(const bool reset) {
    bool retval = true;

    retval = OSPRayArrowGeometry::checkState(this->paramBaseRadius, reset)
        && retval;  // "&& retval" must be last!
    retval = OSPRayArrowGeometry::checkState(this->paramScale, reset)
        && retval;  // "&& retval" must be last!
    retval = OSPRayArrowGeometry::checkState(this->paramTipLength, reset)
        && retval;  // "&& retval" must be last!
    retval = OSPRayArrowGeometry::checkState(this->paramTipRadius, reset)
        && retval;  // "&& retval" must be last!

    return retval;
}


/*
 * megamol::ospray::OSPRayArrowGeometry::create
 */
bool megamol::ospray::OSPRayArrowGeometry::create(void) {
    return true;
}


/*
 * megamol::ospray::OSPRayArrowGeometry::getData
 */
bool megamol::ospray::OSPRayArrowGeometry::getData(core::Call& call) {
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
 * megamol::ospray::OSPRayArrowGeometry::onGetData
 */
bool megamol::ospray::OSPRayArrowGeometry::onGetData(
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
        && this->checkState(true);
    if (isStateChanged) {
        ++this->hashState;
    }

    // Determine whether the data have changed or abort.
    if ((this->hashInput != gd->DataHash()) || isStateChanged) {
        this->frameID = frameTime;
        this->hashInput = gd->DataHash();
    } else {
        // Nothing to do here ...
        return true;
    }

    if (!(*gd)(0)) {
        Log::DefaultLog.WriteError(_T("Failed to get data from source."),
            nullptr);
        return false;
    }

    std::vector<OSPGeometry> geo;
    geo.reserve(gd->GetParticleListCount());

    for (unsigned int i = 0; i <  gd->GetParticleListCount(); ++i) {
        auto& particles = gd->AccessParticles(i);

        if (OSPRayArrowGeometry::checkParticles(particles)) {
            geo.push_back(::ospNewGeometry("arrows"));

            // The data pointer is the first of all relevant pointers, which
            // need to designate the same contiguous memory block if they have
            // passed the particle check above.
            auto vertices = static_cast<const std::int8_t *>(
                particles.GetVertexData());
            auto directions = static_cast<const std::int8_t *>(
                particles.GetDirData());
            auto data = std::min(vertices, directions);

            // Pass the data to OSPRay.
            {
                auto ospData = ::ospNewData(particles.GetCount(),
                    OSPDataType::OSP_DATA,
                    data,
                    OSPDataCreationFlags::OSP_DATA_SHARED_BUFFER);
                ::ospCommit(ospData);
                ::ospSetData(geo.back(), "arrows", ospData);
            }

            //// set bbox
            //auto bboxData = ::ospNewData(6, OSP_FLOAT, particles.GetBBox().PeekBounds());
            //::ospCommit(bboxData);

            // Set parameters of OSPRay arrow geometry.
            ::ospSet1i(geo.back(), "arrow_stride",
                particles.GetVertexDataStride());
            ::ospSet1f(geo.back(), "radius",
                this->paramBaseRadius.Param<FloatParam>()->Value());
            // colours
            // material_id
            ::ospSet1i(geo.back(), "offset_axis",
                directions - data);
            ::ospSet1i(geo.back(), "offset_base",
                vertices - data);
            // "offset_radius" is not supported by MegaMol.
            // offset_colour_id

            switch (particles.GetVertexDataType()) {
                case ParticleType::VertexDataType::VERTDATA_FLOAT_XYZR:
                    ::ospSet1i(geo.back(), "offset_length",
                        vertices - data + 3 * sizeof(float));
                    break;

                default:
                    ::ospSet1i(geo.back(), "offset_length", -1);
                    break;
            }

            // offset_material_id
            // "offset_tip_radius" is not supported by MegaMol.
            ::ospSet1f(geo.back(), "scale",
                this->paramScale.Param<FloatParam>()->Value());
            // "texcoords" is not supported by MegaMol.
            ::ospSet1f(geo.back(), "tip_radius",
                this->paramTipRadius.Param<FloatParam>()->Value());
            ::ospSet1f(geo.back(), "tip_length",
                this->paramTipLength.Param<FloatParam>()->Value());

           // ospSet1i(geo.back(), "colorType", colorType);
            
            // ospSetData(geo.back(), "bbox", bboxData);
            ::ospCommit(geo.back());
        }
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
 * megamol::ospray::OSPRayArrowGeometry::onGetDirty
 */
bool megamol::ospray::OSPRayArrowGeometry::onGetDirty(
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
 * megamol::ospray::OSPRayArrowGeometry::onGetExtents
 */
bool megamol::ospray::OSPRayArrowGeometry::onGetExtents(
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
 * megamol::ospray::OSPRayArrowGeometry::release
 */
void megamol::ospray::OSPRayArrowGeometry::release(void) { }


// void megamol::ospray::OSPRaySphereGeometry::colorTransferGray(
//    std::vector<float>& grayArray, float const* transferTable, unsigned int tableSize, std::vector<float>& rgbaArray)
//    {
//
//    float gray_max = *std::max_element(grayArray.begin(), grayArray.end());
//    float gray_min = *std::min_element(grayArray.begin(), grayArray.end());
//
//    for (auto &gray : grayArray) {
//        float scaled_gray;
//        if ((gray_max - gray_min) <= 1e-4f) {
//            scaled_gray = 0;
//        } else {
//            scaled_gray = (gray - gray_min) / (gray_max - gray_min);
//        }
//        if (transferTable == NULL && tableSize == 0) {
//            for (int i = 0; i < 3; i++) {
//                rgbaArray.push_back((0.3f + scaled_gray) / 1.3f);
//            }
//            rgbaArray.push_back(1.0f);
//        } else {
//            float exact_tf = (tableSize - 1) * scaled_gray;
//            int floor = std::floor(exact_tf);
//            float tail = exact_tf - (float)floor;
//            floor *= 4;
//            for (int i = 0; i < 4; i++) {
//                float colorFloor = transferTable[floor + i];
//                float colorCeil = transferTable[floor + i + 4];
//                float finalColor = colorFloor + (colorCeil - colorFloor)*(tail);
//                rgbaArray.push_back(finalColor);
//            }
//        }
//    }
//}

///*
//* ospray::OSPRaySphereGeometry::getClipData
//*/
// void megamol::ospray::OSPRaySphereGeometry::getClipData(float* clipDat, float* clipCol) {
//    megamol::core::view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<megamol::core::view::CallClipPlane>();
//    if ((ccp != NULL) && (*ccp)()) {
//        clipDat[0] = ccp->GetPlane().Normal().X();
//        clipDat[1] = ccp->GetPlane().Normal().Y();
//        clipDat[2] = ccp->GetPlane().Normal().Z();
//        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
//        clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
//        clipCol[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
//        clipCol[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
//        clipCol[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;
//        clipCol[3] = static_cast<float>(ccp->GetColour()[3]) / 255.0f;
//
//    } else {
//        clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
//        clipCol[0] = clipCol[1] = clipCol[2] = 0.75f;
//        clipCol[3] = 1.0f;
//    }
//}
//
