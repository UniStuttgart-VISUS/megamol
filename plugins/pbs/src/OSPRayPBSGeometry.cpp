/*
* OSPRayPBSGeometry.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayPBSGeometry.h"
#include "vislib/forceinline.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/EnumParam.h"
#include "vislib/sys/Log.h"
#include "mmcore/Call.h"

#include "pbs/PBSDataCall.h"

using namespace megamol::ospray;
using namespace megamol::pbs;


OSPRayPBSGeometry::OSPRayPBSGeometry(void) :
    AbstractOSPRayStructure(),
    getDataSlot("getdata", "Connects to the data source"),
    radiusSlot("radius", "Set the radius for the particles")
{
    this->getDataSlot.SetCompatibleCall<PBSDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);
    
    this->radiusSlot << new core::param::FloatParam(0.1f);
    this->MakeSlotAvailable(&this->radiusSlot);
}


bool OSPRayPBSGeometry::readData(megamol::core::Call &call) {

    // fill material container
    this->processMaterial();

    PBSDataCall *cd = this->getDataSlot.CallAs<PBSDataCall>();

    this->structureContainer.dataChanged = false;
    if (cd == NULL) return false;
    if (this->datahash != cd->DataHash() || this->InterfaceIsDirty()) {
        this->datahash = cd->DataHash();
        this->structureContainer.dataChanged = true;
    } else {
        return true;
    }

    if (!(*cd)(1)) return false;
    if (!(*cd)(0)) return false;

    auto data = cd->GetData().lock();

    auto x_data = data->GetX().lock();
    auto y_data = data->GetY().lock();
    auto z_data = data->GetZ().lock();

    if (x_data->size() != y_data->size() || x_data->size() != z_data->size() || y_data->size() != z_data->size()) {
        return false;
    }

    x_data_float = std::vector<float>(x_data->begin(), x_data->end());
    y_data_float = std::vector<float>(y_data->begin(), y_data->end());
    z_data_float = std::vector<float>(z_data->begin(), z_data->end());


    unsigned int partCount = x_data->size();
    float globalRadius = radiusSlot.Param<core::param::FloatParam>()->Value();

    // Write stuff into the structureContainer
    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::PBS;
    this->structureContainer.xData = std::make_shared<std::vector<float>>(std::move(x_data_float));
    this->structureContainer.yData = std::make_shared<std::vector<float>>(std::move(y_data_float));
    this->structureContainer.zData = std::make_shared<std::vector<float>>(std::move(z_data_float));
    this->structureContainer.partCount = partCount;
    this->structureContainer.globalRadius = globalRadius;

    return true;
}




OSPRayPBSGeometry::~OSPRayPBSGeometry() {
    this->Release();
}

bool OSPRayPBSGeometry::create() {
    return true;
}

void OSPRayPBSGeometry::release() {

}

/*
ospray::OSPRayPBSGeometry::InterfaceIsDirty()
*/
bool OSPRayPBSGeometry::InterfaceIsDirty() {
    if (
        this->radiusSlot.IsDirty()
        ) {
        this->radiusSlot.ResetDirty();
        return true;
    } else {
        return false;
    }
}



bool OSPRayPBSGeometry::getExtends(megamol::core::Call &call) {
    CallOSPRayStructure *os = dynamic_cast<CallOSPRayStructure*>(&call);
    PBSDataCall *cd = this->getDataSlot.CallAs<PBSDataCall>();

    if (cd == NULL) return false;
    cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
                                         // if (!(*cd)(1)) return false; // floattable returns flase at first attempt and breaks everything
    (*cd)(1);
    (*cd)(0);
    // local bounding box is not valid if more than one chunk is loaded!
    //auto localBB = cd->GetLocalBBox().lock();
    //vislib::math::Cuboid<float> cuboid(static_cast<float>(localBB.get()[0]),
    //                                 static_cast<float>(localBB.get()[1]),
    //                                 static_cast<float>(localBB.get()[2]),
    //                                 static_cast<float>(localBB.get()[3]),
    //                                 static_cast<float>(localBB.get()[4]),
    //                                 static_cast<float>(localBB.get()[5]));
    auto globalBB = cd->GetGlobalBBox().lock();
    vislib::math::Cuboid<float> cuboid(static_cast<float>(globalBB.get()[0]),
                                       static_cast<float>(globalBB.get()[1]),
                                       static_cast<float>(globalBB.get()[2]),
                                       static_cast<float>(globalBB.get()[3]),
                                       static_cast<float>(globalBB.get()[4]),
                                       static_cast<float>(globalBB.get()[5]));
    megamol::core::BoundingBoxes bbox;
    bbox.Clear();
    bbox.SetObjectSpaceBBox(cuboid);
    bbox.SetObjectSpaceClipBox(cuboid);
    this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes>(std::move(bbox));
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}