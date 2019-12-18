/*
 * OSPRayLineGeometry.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "OSPRayLineGeometry.h"
#include "OSPRay_plugin/CallOSPRayStructure.h"
#include "geometry_calls/LinesDataCall.h"
#include "mesh/MeshCalls.h"
#include "mmcore/Call.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "vislib/sys/Log.h"

using namespace megamol::ospray;
using namespace megamol;


OSPRayLineGeometry::OSPRayLineGeometry(void)
    : AbstractOSPRayStructure()
    , getDataSlot("getdata", "Connects to the data source")
    , getLineDataSlot("getLineData", "")
    , globalRadiusSlot("globalRadius", "Sets the radius of the lines")
    , smoothSlot("smooth", "Set whether to smooth the lines") {

    this->getDataSlot.SetCompatibleCall<geocalls::LinesDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getLineDataSlot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->getLineDataSlot);

    this->globalRadiusSlot << new core::param::FloatParam(0.01);
    this->MakeSlotAvailable(&this->globalRadiusSlot);

    this->smoothSlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->smoothSlot);
}


bool OSPRayLineGeometry::readData(core::Call& call) {

    // fill material container
    this->processMaterial();

    // get transformation parameter
    this->processTransformation();

    // read Data, calculate  shape parameters, fill data vectors
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);

    mesh::CallMesh* cm = this->getLineDataSlot.CallAs<mesh::CallMesh>();

    if (cm != nullptr) {
        auto meta_data = cm->getMetaData();
        this->structureContainer.dataChanged = false;
        if (os->getTime() > meta_data.m_frame_cnt) {
            meta_data.m_frame_ID = meta_data.m_frame_cnt - 1;
        } else {
            meta_data.m_frame_ID = os->getTime();
        }
        cm->setMetaData(meta_data);
        if (!(*cm)(1)) return false;
        if (!(*cm)(0)) return false;
        meta_data = cm->getMetaData();

        if (cm->hasUpdate() || this->time != os->getTime() || this->InterfaceIsDirty()) {
            this->time = os->getTime();
            this->structureContainer.dataChanged = true;
            this->extendContainer.boundingBox = std::make_shared<core::BoundingBoxes_2>(meta_data.m_bboxs);
            this->structureContainer.mesh = cm->getData();
        }

    } else {

        geocalls::LinesDataCall* cd = this->getDataSlot.CallAs<geocalls::LinesDataCall>();
        this->structureContainer.dataChanged = false;
        if (cd == NULL) return false;
        cd->SetTime(os->getTime());
        cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
        if (this->datahash != cd->DataHash() || this->time != os->getTime() || this->InterfaceIsDirty()) {
            this->datahash = cd->DataHash();
            this->time = os->getTime();
            this->structureContainer.dataChanged = true;
        } else {
            return true;
        }

        if (!(*cd)(1)) return false;
        if (!(*cd)(0)) return false;

        unsigned int lineCount = cd->Count();

        std::vector<float> vd;
        std::vector<float> cd_rgba;
        std::vector<unsigned int> index;

        unsigned int indexWalker = 0;

        // Generate vertex and index arrays
        for (unsigned int i = 0; i < lineCount; ++i) {
            auto line = cd->GetLines()[i];
            for (unsigned int j = 0; j < line.Count(); ++j) {
                vd.push_back(line.VertexArrayFloat()[3 * j + 0]);
                vd.push_back(line.VertexArrayFloat()[3 * j + 1]);
                vd.push_back(line.VertexArrayFloat()[3 * j + 2]);
                if (line.ColourArrayType() == geocalls::LinesDataCall::Lines::CDT_FLOAT_RGB) {
                    cd_rgba.push_back(line.ColourArrayFloat()[3 * j + 0]);
                    cd_rgba.push_back(line.ColourArrayFloat()[3 * j + 1]);
                    cd_rgba.push_back(line.ColourArrayFloat()[3 * j + 2]);
                    cd_rgba.push_back(1.0f);
                } else if (line.ColourArrayType() == geocalls::LinesDataCall::Lines::CDT_FLOAT_RGBA) {
                    cd_rgba.push_back(line.ColourArrayFloat()[4 * j + 0]);
                    cd_rgba.push_back(line.ColourArrayFloat()[4 * j + 1]);
                    cd_rgba.push_back(line.ColourArrayFloat()[4 * j + 2]);
                    cd_rgba.push_back(line.ColourArrayFloat()[4 * j + 3]);
                } else if (line.ColourArrayType() == geocalls::LinesDataCall::Lines::CDT_BYTE_RGB) {
                    cd_rgba.push_back((static_cast<unsigned int>(line.ColourArrayByte()[0]) / 255.0f));
                    cd_rgba.push_back((static_cast<unsigned int>(line.ColourArrayByte()[1]) / 255.0f));
                    cd_rgba.push_back((static_cast<unsigned int>(line.ColourArrayByte()[2]) / 255.0f));
                    cd_rgba.push_back(1.0f);
                } else if (line.ColourArrayType() == geocalls::LinesDataCall::Lines::CDT_BYTE_RGBA) {
                    cd_rgba.push_back((static_cast<unsigned int>(line.ColourArrayByte()[0]) / 255.0f));
                    cd_rgba.push_back((static_cast<unsigned int>(line.ColourArrayByte()[1]) / 255.0f));
                    cd_rgba.push_back((static_cast<unsigned int>(line.ColourArrayByte()[2]) / 255.0f));
                    cd_rgba.push_back((static_cast<unsigned int>(line.ColourArrayByte()[3]) / 255.0f));
                }
                index.push_back(indexWalker);
                indexWalker += 1;
            }
            index.pop_back();
        }

        if (cd->GetLines()->ColourArrayType() == geocalls::LinesDataCall::Lines::CDT_NONE) {
            auto col = cd->GetLines()->GlobalColour();
            float r = static_cast<unsigned int>(col.R()) / 255.0f;
            float g = static_cast<unsigned int>(col.G()) / 255.0f;
            float b = static_cast<unsigned int>(col.B()) / 255.0f;
            float a = static_cast<unsigned int>(col.A()) / 255.0f;
            for (unsigned int i = 0; i < vd.size(); ++i) {
                cd_rgba.push_back(r);
                cd_rgba.push_back(g);
                cd_rgba.push_back(b);
                cd_rgba.push_back(a);
            }
        }


        // Write stuff into the structureContainer

        this->structureContainer.vertexData = std::make_shared<std::vector<float>>(std::move(vd));
        this->structureContainer.colorData = std::make_shared<std::vector<float>>(std::move(cd_rgba));
        this->structureContainer.vertexLength = 3;
        this->structureContainer.colorLength = 4;
        this->structureContainer.indexData = std::make_shared<std::vector<unsigned int>>(std::move(index));
    }

    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::STREAMLINES;
    this->structureContainer.globalRadius = globalRadiusSlot.Param<core::param::FloatParam>()->Value();
    this->structureContainer.smooth = smoothSlot.Param<core::param::BoolParam>()->Value();

    return true;
}

OSPRayLineGeometry::~OSPRayLineGeometry() { this->Release(); }

bool OSPRayLineGeometry::create() { return true; }

void OSPRayLineGeometry::release() {}

/*
ospray::OSPRayLineGeometry::InterfaceIsDirty()
*/
bool OSPRayLineGeometry::InterfaceIsDirty() {
    CallOSPRayMaterial* cm = this->getMaterialSlot.CallAs<CallOSPRayMaterial>();
    cm->getMaterialParameter();
    if (cm->InterfaceIsDirty() || this->globalRadiusSlot.IsDirty() || this->smoothSlot.IsDirty()) {
        this->globalRadiusSlot.ResetDirty();
        this->smoothSlot.ResetDirty();
        return true;
    } else {
        return false;
    }
}


bool OSPRayLineGeometry::getExtends(core::Call& call) {
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);

    mesh::CallMesh* cm = this->getLineDataSlot.CallAs<mesh::CallMesh>();
    if (cm != nullptr) {

        if (!(*cm)(1)) return false;
        auto meta_data = cm->getMetaData();
        if (os->getTime() > meta_data.m_frame_cnt) {
            meta_data.m_frame_ID = meta_data.m_frame_cnt - 1;
        } else {
            meta_data.m_frame_ID = os->getTime();
        }
        cm->setMetaData(meta_data);

        this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes_2>(meta_data.m_bboxs);
        this->extendContainer.timeFramesCount = meta_data.m_frame_cnt;
        this->extendContainer.isValid = true;

    } else {
        geocalls::LinesDataCall* cd = this->getDataSlot.CallAs<geocalls::LinesDataCall>();

        if (cd == NULL) return false;
        cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
        // if (!(*cd)(1)) return false; // table returns flase at first attempt and breaks everything
        (*cd)(1);
        this->extendContainer.boundingBox = std::make_shared<core::BoundingBoxes_2>();
        this->extendContainer.boundingBox->SetBoundingBox(cd->AccessBoundingBoxes().ObjectSpaceBBox());
        this->extendContainer.timeFramesCount = cd->FrameCount();
        this->extendContainer.isValid = true;
    }

    return true;
}