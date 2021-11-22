/*
 * OSPRayGeometryTest.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "OSPRayGeometryTest.h"
#include "mmcore/Call.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"

using namespace megamol::ospray;

OSPRayGeometryTest::OSPRayGeometryTest(void) : AbstractOSPRayStructure() {}


bool OSPRayGeometryTest::readData(megamol::core::Call& call) {

    // fill material container
    this->processMaterial();

    // get transformation parameter
    this->processTransformation();

    // read Data, calculate  shape parameters, fill data vectors

    this->structureContainer.dataChanged = true;


    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::TEST;

    return true;
}


OSPRayGeometryTest::~OSPRayGeometryTest() {
    this->Release();
}

bool OSPRayGeometryTest::create() {
    return true;
}

void OSPRayGeometryTest::release() {}

/*
ospray::OSPRayGeometryTest::InterfaceIsDirty()
*/
bool OSPRayGeometryTest::InterfaceIsDirty() {
    return false;
}


bool OSPRayGeometryTest::getExtends(megamol::core::Call& call) {
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);

    this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes_2>();
    this->extendContainer.boundingBox->SetBoundingBox(-1, -1, 1, 1, 1, -1);
    this->extendContainer.timeFramesCount = 1;
    this->extendContainer.isValid = true;

    return true;
}
