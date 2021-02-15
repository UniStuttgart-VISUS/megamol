/*
 * OSPRaySphereGeometry.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "OSPRaySphereGeometry.h"
#include "mmcore/Call.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/forceinline.h"

#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"


using namespace megamol::ospray;


OSPRaySphereGeometry::OSPRaySphereGeometry(void)
        : AbstractOSPRayStructure()
        , getDataSlot("getdata", "Connects to the data source")
        , getClipPlaneSlot("getclipplane", "Connects to a clipping plane module")
        , particleList("ParticleList", "Switches between particle lists") {

    this->getDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getClipPlaneSlot.SetCompatibleCall<core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);

    this->particleList << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->particleList);
}


bool OSPRaySphereGeometry::readData(megamol::core::Call& call) {

    // fill material container
    this->processMaterial();

    // fill transformation container
    this->processTransformation();

    // read Data, calculate  shape parameters, fill data vectors
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::core::moldyn::MultiParticleDataCall* cd =
        this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();
    if (cd == nullptr)
        return false;

    cd->SetFrameID(os->getTime(), true);
    if (!(*cd)(1))
        return false;
    if (!(*cd)(0))
        return false;

    if (this->datahash != cd->DataHash() || this->time != cd->FrameID() || this->InterfaceIsDirty()) {
        if (cd->GetParticleListCount() == 0)
            return false;

        if (this->particleList.Param<core::param::IntParam>()->Value() > (cd->GetParticleListCount() - 1)) {
            this->particleList.Param<core::param::IntParam>()->SetValue(0);
        }

        core::moldyn::MultiParticleDataCall::Particles& parts =
            cd->AccessParticles(this->particleList.Param<core::param::IntParam>()->Value());

        auto const partCount = parts.GetCount();
        auto const globalRadius = parts.GetGlobalRadius();

        vd.resize(partCount * 3ul);
        cd_rgba.resize(partCount * 4ul);

        auto const xAcc = parts.GetParticleStore().GetXAcc();
        auto const yAcc = parts.GetParticleStore().GetYAcc();
        auto const zAcc = parts.GetParticleStore().GetZAcc();

        if (parts.GetColourDataType() != core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I &&
            parts.GetColourDataType() != core::moldyn::SimpleSphericalParticles::COLDATA_DOUBLE_I) {
            auto const crAcc = parts.GetParticleStore().GetCRAcc();
            auto const cgAcc = parts.GetParticleStore().GetCGAcc();
            auto const cbAcc = parts.GetParticleStore().GetCBAcc();
            auto const caAcc = parts.GetParticleStore().GetCAAcc();

            for (std::size_t pidx = 0; pidx < partCount; ++pidx) {
                vd[pidx * 3 + 0] = xAcc->Get_f(pidx);
                vd[pidx * 3 + 1] = yAcc->Get_f(pidx);
                vd[pidx * 3 + 2] = zAcc->Get_f(pidx);

                cd_rgba[pidx * 4 + 0] = crAcc->Get_f(pidx);
                cd_rgba[pidx * 4 + 1] = cgAcc->Get_f(pidx);
                cd_rgba[pidx * 4 + 2] = cbAcc->Get_f(pidx);
                cd_rgba[pidx * 4 + 3] = caAcc->Get_f(pidx);
            }
        } else {
            core::utility::log::Log::DefaultLog.WriteWarn(
                "[OSPRaySphereGeometry]: Color type not supported. Fallback to constant color.");

            auto const g_color = parts.GetGlobalColour();

            for (std::size_t pidx = 0; pidx < partCount; ++pidx) {
                vd[pidx * 3 + 0] = xAcc->Get_f(pidx);
                vd[pidx * 3 + 1] = yAcc->Get_f(pidx);
                vd[pidx * 3 + 2] = zAcc->Get_f(pidx);

                cd_rgba[pidx * 4 + 0] = g_color[0] / 255.0f;
                cd_rgba[pidx * 4 + 1] = g_color[1] / 255.0f;
                cd_rgba[pidx * 4 + 2] = g_color[2] / 255.0f;
                cd_rgba[pidx * 4 + 3] = g_color[3] / 255.0f;
            }
        }

        sphereStructure ss;

        ss.vertexData = std::make_shared<std::vector<float>>(std::move(vd));
        ss.colorData = std::make_shared<std::vector<float>>(std::move(cd_rgba));
        ss.vertexLength = 3;
        ss.colorLength = 4;
        ss.partCount = partCount;
        ss.globalRadius = globalRadius;

        this->structureContainer.structure = ss;

        this->datahash = cd->DataHash();
        this->time = cd->FrameID();
        this->structureContainer.dataChanged = true;
    } else {
        this->structureContainer.dataChanged = false;
    }

    // clipPlane setup
    std::array<float,4> clipDat;
    std::array<float,4> clipCol;
    this->getClipData(clipDat.data(), clipCol.data());


    // Write stuff into the structureContainer
    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::SPHERES;
    //this->structureContainer.clipPlaneData = std::make_shared<std::vector<float>>(std::move(clipDat));
    //this->structureContainer.clipPlaneColor = std::make_shared<std::vector<float>>(std::move(clipCol));

    return true;
}


OSPRaySphereGeometry::~OSPRaySphereGeometry() {
    this->Release();
}


bool OSPRaySphereGeometry::create() {
    return true;
}


void OSPRaySphereGeometry::release() {}


bool OSPRaySphereGeometry::InterfaceIsDirty() {
    if (this->particleList.IsDirty()) {
        this->particleList.ResetDirty();
        return true;
    } else {
        return false;
    }
}


void OSPRaySphereGeometry::getClipData(float* clipDat, float* clipCol) {
    megamol::core::view::CallClipPlane* ccp = this->getClipPlaneSlot.CallAs<megamol::core::view::CallClipPlane>();
    if ((ccp != NULL) && (*ccp)()) {
        clipDat[0] = ccp->GetPlane().Normal().X();
        clipDat[1] = ccp->GetPlane().Normal().Y();
        clipDat[2] = ccp->GetPlane().Normal().Z();
        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
        clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
        clipCol[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
        clipCol[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
        clipCol[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;
        clipCol[3] = static_cast<float>(ccp->GetColour()[3]) / 255.0f;

    } else {
        clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
        clipCol[0] = clipCol[1] = clipCol[2] = 0.75f;
        clipCol[3] = 1.0f;
    }
}


bool OSPRaySphereGeometry::getExtends(megamol::core::Call& call) {
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::core::moldyn::MultiParticleDataCall* cd =
        this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();

    if (cd == NULL)
        return false;
    cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    // if (!(*cd)(1)) return false; // table returns flase at first attempt and breaks everything
    (*cd)(1);
    this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes_2>();
    this->extendContainer.boundingBox->SetBoundingBox(cd->AccessBoundingBoxes().ObjectSpaceBBox());
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}
