/*
 * AbstractSphereRenderer.cpp
 *
 * Copyright (C) 2012 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */


#include "stdafx.h"
#include "mmcore/moldyn/AbstractSphereRenderer.h"


using namespace megamol::core;


/*
 * moldyn::AbstractSphereRenderer::AbstractSphereRenderer
 */
moldyn::AbstractSphereRenderer::AbstractSphereRenderer(void) : Renderer3DModule(),
        getDataSlot("getdata", "Connects to the data source"),
        getTFSlot("gettransferfunction", "Connects to the transfer function module"),
        getClipPlaneSlot("getclipplane", "Connects to a clipping plane module"),
        greyTF(0),
        forceTimeSlot("forceTime", "Flag to force the time code to the specified value. Set to true when rendering a video."),
        useLocalBBoxParam("useLocalBBox", "Enforce usage of local bbox for camera setup") {

    this->getDataSlot.SetCompatibleCall<moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getClipPlaneSlot.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);

    this->forceTimeSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->forceTimeSlot);

    this->useLocalBBoxParam << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->useLocalBBoxParam);
}


/*
 * moldyn::AbstractSphereRenderer::~AbstractSphereRenderer
 */
moldyn::AbstractSphereRenderer::~AbstractSphereRenderer(void) {
    this->Release();
}


/*
 * moldyn::AbstractSphereRenderer::create
 */
bool moldyn::AbstractSphereRenderer::create(void) {
    glEnable(GL_TEXTURE_1D);
    glGenTextures(1, &this->greyTF);
    unsigned char tex[6] = {
        0, 0, 0,  255, 255, 255
    };
    glBindTexture(GL_TEXTURE_1D, this->greyTF);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glBindTexture(GL_TEXTURE_1D, 0);

    glDisable(GL_TEXTURE_1D);

    return true;
}


/*
 * moldyn::AbstractSphereRenderer::GetExtents
 */
bool moldyn::AbstractSphereRenderer::GetExtents(view::CallRender3D& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    MultiParticleDataCall *c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if ((c2 != NULL)) {
        c2->SetFrameID(static_cast<unsigned int>(cr->Time()), this->isTimeForced());
        if (!(*c2)(1)) return false;
        cr->SetTimeFramesCount(c2->FrameCount());
        auto const plcount = c2->GetParticleListCount();
        if (this->useLocalBBoxParam.Param<param::BoolParam>()->Value() && plcount > 0) {
            auto bbox = c2->AccessParticles(0).GetBBox();
            auto cbbox = bbox;
            cbbox.Grow(c2->AccessParticles(0).GetGlobalRadius());
            for (unsigned pidx = 1; pidx < plcount; ++pidx) {
                auto temp = c2->AccessParticles(pidx).GetBBox();
                bbox.Union(temp);
                temp.Grow(c2->AccessParticles(pidx).GetGlobalRadius());
                cbbox.Union(temp);
            }
            cr->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
            cr->AccessBoundingBoxes().SetObjectSpaceClipBox(cbbox);
        } else {
            cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();
        }

        float scaling = cr->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        } else {
            scaling = 1.0f;
        }
        cr->AccessBoundingBoxes().MakeScaledWorld(scaling);
    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }

    return true;
}


/*
 * moldyn::AbstractSphereRenderer::release
 */
void moldyn::AbstractSphereRenderer::release(void) {
    ::glDeleteTextures(1, &this->greyTF);
}


/*
 * moldyn::AbstractSphereRenderer::getData
 */
moldyn::MultiParticleDataCall *moldyn::AbstractSphereRenderer::getData(unsigned int t, float& outScaling) {
    MultiParticleDataCall *c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
    outScaling = 1.0f;
    if (c2 != NULL) {
        c2->SetFrameID(t, this->isTimeForced());
        if (!(*c2)(1)) return NULL;

        // calculate scaling
        auto const plcount = c2->GetParticleListCount();
        if (this->useLocalBBoxParam.Param<param::BoolParam>()->Value() && plcount > 0) {
            outScaling = c2->AccessParticles(0).GetBBox().LongestEdge();
            for (unsigned pidx = 0; pidx < plcount; ++pidx) {
                auto const temp = c2->AccessParticles(pidx).GetBBox().LongestEdge();
                if (outScaling < temp) {
                    outScaling = temp;
                }
            }
        } else {
            outScaling = c2->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        }
        if (outScaling > 0.0000001) {
            outScaling = 10.0f / outScaling;
        } else {
            outScaling = 1.0f;
        }

        c2->SetFrameID(t, this->isTimeForced());
        if (!(*c2)(0)) return NULL;

        return c2;
    } else {
        return NULL;
    }
}


/*
 * moldyn::AbstractSphereRenderer::getClipData
 */
void moldyn::AbstractSphereRenderer::getClipData(float *clipDat, float *clipCol) {
    view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<view::CallClipPlane>();
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


bool moldyn::AbstractSphereRenderer::isTimeForced(void) const {
    return this->forceTimeSlot.Param<param::BoolParam>()->Value();
}
