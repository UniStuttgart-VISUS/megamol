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

    unsigned int partCount = x_data->size();
    float globalRadius = radiusSlot.Param<core::param::FloatParam>()->Value();

    // Write stuff into the structureContainer
    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::PBS;
    this->structureContainer.xData = std::move(x_data);
    this->structureContainer.yData = std::move(y_data);
    this->structureContainer.zData = std::move(z_data);
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
    this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes>(cd->AccessBoundingBoxes());
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}