/*
 * OSPRayPKDGeometry.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "OSPRayPKDGeometry.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"
#include "vislib/forceinline.h"

#include "mmcore/view/CallClipPlane.h"
#include "mmospray/CallOSPRayAPIObject.h"
#include "ospray/ospray_cpp.h"

namespace megamol {
namespace ospray {


OSPRayPKDGeometry::OSPRayPKDGeometry(void)
        : getDataSlot("getdata", "Connects to the data source")
        , deployStructureSlot("deployStructureSlot", "Connects to an OSPRayAPIStructure")
        , colorTypeSlot("colorType", "Set the type of encoded color") {

    this->getDataSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    auto ep = new megamol::core::param::EnumParam(0);
    ep->SetTypePair(0, "none");
    ep->SetTypePair(1, "RGBu8");
    ep->SetTypePair(2, "RGBAu8");
    ep->SetTypePair(3, "RGBf");
    ep->SetTypePair(4, "RGBAf");
    ep->SetTypePair(5, "I");
    this->colorTypeSlot << ep;
    this->MakeSlotAvailable(&this->colorTypeSlot);

    this->deployStructureSlot.SetCallback(
        CallOSPRayAPIObject::ClassName(), CallOSPRayAPIObject::FunctionName(0), &OSPRayPKDGeometry::getDataCallback);
    this->deployStructureSlot.SetCallback(
        CallOSPRayAPIObject::ClassName(), CallOSPRayAPIObject::FunctionName(1), &OSPRayPKDGeometry::getExtendsCallback);
    this->deployStructureSlot.SetCallback(
        CallOSPRayAPIObject::ClassName(), CallOSPRayAPIObject::FunctionName(2), &OSPRayPKDGeometry::getDirtyCallback);
    this->MakeSlotAvailable(&this->deployStructureSlot);
}


bool OSPRayPKDGeometry::getDataCallback(megamol::core::Call& call) {

    // read Data, calculate  shape parameters, fill data vectors
    auto os = dynamic_cast<CallOSPRayAPIObject*>(&call);
    megamol::geocalls::MultiParticleDataCall* cd = this->getDataSlot.CallAs<megamol::geocalls::MultiParticleDataCall>();


    //auto const minFrameCount = cd->FrameCount();

    //if (minFrameCount == 0) return false;

    //auto frameTime = 0;

    //if (os->FrameID() >= minFrameCount) {
    //    cd->SetFrameID(minFrameCount - 1, true); // isTimeForced flag set to true
    //    frameTime = minFrameCount - 1;
    //} else {
    //    cd->SetFrameID(os->FrameID(), true); // isTimeForced flag set to true
    //    frameTime = os->FrameID();
    //}
    cd->SetFrameID(os->FrameID(), true);
    if (!(*cd)(1))
        return false;

    if (this->datahash != cd->DataHash() || this->time != cd->FrameID() || this->InterfaceIsDirty()) {
        this->datahash = cd->DataHash();
        this->time = cd->FrameID();
    } else {
        return true;
    }

    if (!(*cd)(0))
        return false;

    size_t listCount = cd->GetParticleListCount();
    std::vector<::ospray::cpp::Geometry> geo;
    for (size_t i = 0; i < listCount; i++) {

        geocalls::MultiParticleDataCall::Particles& parts = cd->AccessParticles(i);

        auto colorType = this->colorTypeSlot.Param<megamol::core::param::EnumParam>()->Value();

        geo.emplace_back(ospNewGeometry("pkd_geometry"));

        auto vertexData =
            ::ospray::cpp::SharedData(parts.GetVertexData(), OSP_FLOAT, parts.GetCount(), 4 * sizeof(float));
        vertexData.commit();

        // set bbox
        auto bboxData = ::ospray::cpp::CopiedData(parts.GetBBox().PeekBounds(), OSP_FLOAT, 6);
        bboxData.commit();

        geo.back().setParam("radius", parts.GetGlobalRadius());
        //ospSet1i(geo.back(), "colorType", colorType);
        geo.back().setParam("colorType", 2);
        geo.back().setParam("position", vertexData);
        // ospSetData(geo.back(), "bbox", bboxData);
        geo.back().setParam("bbox", NULL);
        geo.back().commit();

        // TODO: implement distributed stuff
        // if (this->rd_type.Param<megamol::core::param::EnumParam>()->Value() == MPI_RAYCAST) {
        //    auto const half_radius = element.globalRadius * 0.5f;

        //    auto const bbox = element.boundingBox->ObjectSpaceBBox().PeekBounds(); //< TODO Not all geometries expose
        //    bbox ospcommon::vec3f lower{bbox[0] - half_radius, bbox[1] - half_radius,
        //        bbox[2] - half_radius}; //< TODO The bbox needs to include complete sphere bound
        //    ospcommon::vec3f upper{bbox[3] + half_radius, bbox[4] + half_radius, bbox[5] + half_radius};
        //    // ghostRegions.emplace_back(lower, upper);
        //    worldBounds.extend({lower, upper}); //< TODO Possible hazard if bbox is not centered
        //    regions.emplace_back(lower, upper);
        //}
    }


    std::vector<void*> geo_transfer(geo.size());
    for (auto i = 0; i < geo.size(); i++) {
        geo_transfer[i] = geo[i].handle();
    }
    os->setStructureType(GEOMETRY);
    os->setAPIObjects(std::move(geo_transfer));

    return true;
}


OSPRayPKDGeometry::~OSPRayPKDGeometry() {
    this->Release();
}

bool OSPRayPKDGeometry::create() {
    auto error = ospLoadModule("pkd");
    if (error != OSPError::OSP_NO_ERROR) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to load OSPRay module: PKD. Error occured in %s:%d", __FILE__, __LINE__);
    }
    return true;
}

void OSPRayPKDGeometry::release() {}

/*
ospray::OSPRayPKDGeometry::InterfaceIsDirty()
*/
bool OSPRayPKDGeometry::InterfaceIsDirty() {
    if (this->colorTypeSlot.IsDirty()) {
        this->colorTypeSlot.ResetDirty();
        return true;
    } else {
        return false;
    }
}

bool OSPRayPKDGeometry::InterfaceIsDirtyNoReset() const {
    return this->colorTypeSlot.IsDirty();
}


bool OSPRayPKDGeometry::getExtendsCallback(core::Call& call) {
    auto os = dynamic_cast<CallOSPRayAPIObject*>(&call);
    geocalls::MultiParticleDataCall* cd = this->getDataSlot.CallAs<geocalls::MultiParticleDataCall>();

    if (cd == NULL)
        return false;
    cd->SetFrameID(os->FrameID(), true); // isTimeForced flag set to true
    // if (!(*cd)(1)) return false; // table returns flase at first attempt and breaks everything
    (*cd)(1);

    core::BoundingBoxes_2 box;
    box.SetBoundingBox(cd->AccessBoundingBoxes().ObjectSpaceBBox());

    os->SetExtent(cd->FrameCount(), box);

    return true;
}

bool OSPRayPKDGeometry::getDirtyCallback(core::Call& call) {
    auto os = dynamic_cast<CallOSPRayAPIObject*>(&call);
    auto cd = this->getDataSlot.CallAs<megamol::geocalls::MultiParticleDataCall>();

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


} // namespace ospray
} // namespace megamol
