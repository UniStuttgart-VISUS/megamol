/*
 * DirPartFilter.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "DirPartFilter.h"
#include "param/FloatParam.h"
#include "param/StringParam.h"
//#include "vislib/ColourParser.h"
#include <climits>

using namespace megamol::core;


/*
 * moldyn::DirPartFilter::DirPartFilter
 */
moldyn::DirPartFilter::DirPartFilter(void) : Module(),
        inParticlesDataSlot("inPartData", "Input for oriented particle data"),
        inVolumeDataSlot("inVolData", "Input for volume data"),
        outDataSlot("outData", "Output of oriented particle data"),
        attributeSlot("attr", "The volume attribute to use"),
        attrMinValSlot("attrMinVal", "The minimum value"),
        attrMaxValSlot("attrMaxVal", "The maximum value"),
        datahashOut(0), datahashParticlesIn(0), datahashVolumeIn(0),
        frameID(0), vertData() {

    this->inParticlesDataSlot.SetCompatibleCall<moldyn::DirectionalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inParticlesDataSlot);

    this->inVolumeDataSlot.SetCompatibleCall<CallVolumeDataDescription>();
    this->MakeSlotAvailable(&this->inVolumeDataSlot);

    this->outDataSlot.SetCallback(moldyn::DirectionalParticleDataCall::ClassName(),
        moldyn::DirectionalParticleDataCall::FunctionName(0), &DirPartFilter::getData);
    this->outDataSlot.SetCallback(moldyn::DirectionalParticleDataCall::ClassName(),
        moldyn::DirectionalParticleDataCall::FunctionName(1), &DirPartFilter::getExtend);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->attributeSlot << new param::StringParam("0");
    this->MakeSlotAvailable(&this->attributeSlot);

    this->attrMinValSlot << new param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->attrMinValSlot);

    this->attrMaxValSlot << new param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->attrMaxValSlot);
}


/*
 * moldyn::DirPartFilter::~DirPartFilter
 */
moldyn::DirPartFilter::~DirPartFilter(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * moldyn::DirPartFilter::create
 */
bool moldyn::DirPartFilter::create(void) {
    // intentionally empty
    return true;
}


/*
 * moldyn::DirPartFilter::release
 */
void moldyn::DirPartFilter::release(void) {
    // intentionally empty
}


/*
 * moldyn::DirPartFilter::getData
 */
bool moldyn::DirPartFilter::getData(Call& call) {
    bool rebuildData = false;
    if (this->datahashOut == 0) rebuildData = true;

    DirectionalParticleDataCall *outDpdc = dynamic_cast<DirectionalParticleDataCall*>(&call);
    if (outDpdc == NULL) return false;
    DirectionalParticleDataCall *inDpdc = this->inParticlesDataSlot.CallAs<DirectionalParticleDataCall>();
    if (inDpdc == NULL) return false;
    CallVolumeData *inCvd = this->inVolumeDataSlot.CallAs<CallVolumeData>();
    if (inCvd == NULL) return false;

    if (this->attributeSlot.IsDirty()) {
        rebuildData = true;
        this->attributeSlot.ResetDirty();
    }
    if (this->attrMinValSlot.IsDirty()) {
        rebuildData = true;
        this->attrMinValSlot.ResetDirty();
    }
    if (this->attrMaxValSlot.IsDirty()) {
        rebuildData = true;
        this->attrMaxValSlot.ResetDirty();
    }

    *inDpdc = *outDpdc;
    if (!(*inDpdc)(0)) {
        return false;
    }
    *outDpdc = *inDpdc;
    // We really only support XYZ ATM
    for (int i = outDpdc->GetParticleListCount() - 1; i >= 0; i--) {
        //switch (outDpdc->AccessParticles(i).GetColourDataType()) {
        //    case DirectionalParticleDataCall::Particles::COLDATA_FLOAT_I: // falls through
        //    case DirectionalParticleDataCall::Particles::COLDATA_NONE: // falls through
        //        return false;
        //}
        switch (outDpdc->AccessParticles(i).GetVertexDataType()) {
            case DirectionalParticleDataCall::Particles::VERTDATA_NONE: // falls through
            case DirectionalParticleDataCall::Particles::VERTDATA_SHORT_XYZ: // falls through
            case DirectionalParticleDataCall::Particles::VERTDATA_FLOAT_XYZR: // falls through
                return false;
        }
    }
    if (inDpdc->DataHash() != this->datahashParticlesIn) {
        rebuildData = true;
        this->datahashParticlesIn = inDpdc->DataHash();
    }
    if (outDpdc->FrameID() != this->frameID) {
        rebuildData = true;
    }

    inCvd->SetFrameID(outDpdc->FrameID(), outDpdc->IsFrameForced());
    if (!(*inCvd)(1)) { // just poke for update
        return false;
    }
    vislib::math::Cuboid<float> volBB(inCvd->AccessBoundingBoxes().ObjectSpaceBBox());
    inCvd->SetFrameID(outDpdc->FrameID(), outDpdc->IsFrameForced());
    if (!(*inCvd)(0)) { // just poke for update
        return false;
    }
    if (inCvd->DataHash() != this->datahashVolumeIn) {
        rebuildData = true;
        this->datahashVolumeIn = inCvd->DataHash();
    }

    if (rebuildData) {
        //float bcR, bcG, bcB, r, g, b, v;
        //vislib::graphics::ColourParser::FromString(this->baseColourSlot.Param<param::StringParam>()->Value(), bcR, bcG, bcB);
        vislib::StringA attrName(this->attributeSlot.Param<param::StringParam>()->Value());
        unsigned int attrIdx = inCvd->FindAttribute(attrName);
        if (attrIdx == UINT_MAX) {
            try {
                attrIdx = static_cast<unsigned int>(vislib::CharTraitsA::ParseInt(attrName));
            } catch(...) {
                attrIdx = UINT_MAX;
            }
        }

        if (attrIdx != UINT_MAX) {
            const CallVolumeData::Data& vol = inCvd->Attribute(attrIdx);
            float valMin = this->attrMinValSlot.Param<param::FloatParam>()->Value();
            float valMax = this->attrMaxValSlot.Param<param::FloatParam>()->Value();
            //if (vislib::math::IsEqual(valRng, 0.0f)) valRng = 1.0f;
            SIZE_T cnt = 0;
            unsigned int plCnt = outDpdc->GetParticleListCount();
            for (unsigned int i = 0; i < plCnt; i++) {
                SIZE_T c = static_cast<SIZE_T>(outDpdc->AccessParticles(i).GetCount());
                c *= 4;
                cnt += c;
            }
            this->vertData.EnforceSize(cnt * sizeof(float));
            cnt = 0;
            for (unsigned int i = 0; i < plCnt; i++) {
                SIZE_T pCnt = static_cast<SIZE_T>(outDpdc->AccessParticles(i).GetCount());
                const float *inVertData = static_cast<const float*>(outDpdc->AccessParticles(i).GetVertexData());
                unsigned int inVertStep = outDpdc->AccessParticles(i).GetVertexDataStride();
                ASSERT((inVertStep % sizeof(float)) == 0);
                float inRad = outDpdc->AccessParticles(i).GetGlobalRadius();

                for (SIZE_T p = 0; p < pCnt; p++, cnt += 4 * sizeof(float), inVertData += (inVertStep / sizeof(float))) {
                    float x = (inVertData[0] - volBB.Left()) / volBB.Width();
                    float y = (inVertData[1] - volBB.Bottom()) / volBB.Height();
                    float z = (inVertData[2] - volBB.Back()) / volBB.Depth();
                    float r = (true) ? inRad : inVertData[3];

                    x *= static_cast<float>(inCvd->XSize());
                    y *= static_cast<float>(inCvd->YSize());
                    z *= static_cast<float>(inCvd->ZSize());
                    x -= 0.5f;
                    y -= 0.5f;
                    z -= 0.5f;
                    int ix = static_cast<int>(x);
                    int iy = static_cast<int>(y);
                    int iz = static_cast<int>(z);
                    x -= static_cast<float>(ix);
                    y -= static_cast<float>(iy);
                    z -= static_cast<float>(iz);

                    float vv[8];
                    for (int ox = 0; ox < 2; ox++) {
                        int px = ix + ox;
                        if (px < 0) px = 0;
                        if (px >= static_cast<int>(inCvd->XSize())) px = static_cast<int>(inCvd->XSize()) - 1;
                        for (int oy = 0; oy < 2; oy++) {
                            int py = iy + oy;
                            if (py < 0) py = 0;
                            if (py >= static_cast<int>(inCvd->YSize())) py = static_cast<int>(inCvd->YSize()) - 1;
                            for (int oz = 0; oz < 2; oz++) {
                                int pz = iz + oz;
                                if (pz < 0) pz = 0;
                                if (pz >= static_cast<int>(inCvd->ZSize())) pz = static_cast<int>(inCvd->ZSize()) - 1;
                                switch (vol.Type()) {
                                case CallVolumeData::TYPE_BYTE:
                                    vv[ox + 2 * (oy + 2 * oz)] = static_cast<float>(vol.Bytes()[px + inCvd->XSize() * (py + inCvd->YSize() * pz)]) / 255.0f;
                                    break;
                                case CallVolumeData::TYPE_DOUBLE:
                                    vv[ox + 2 * (oy + 2 * oz)] = static_cast<float>(vol.Doubles()[px + inCvd->XSize() * (py + inCvd->YSize() * pz)]);
                                    break;
                                case CallVolumeData::TYPE_FLOAT:
                                    vv[ox + 2 * (oy + 2 * oz)] = vol.Floats()[px + inCvd->XSize() * (py + inCvd->YSize() * pz)];
                                    break;
                                }
                            }
                        }
                    }

                    vv[0] = (1.0f - x) * vv[0] + x * vv[1];
                    vv[2] = (1.0f - x) * vv[2] + x * vv[3];
                    vv[4] = (1.0f - x) * vv[4] + x * vv[5];
                    vv[6] = (1.0f - x) * vv[6] + x * vv[7];

                    vv[0] = (1.0f - y) * vv[0] + y * vv[2];
                    vv[4] = (1.0f - y) * vv[4] + y * vv[6];

                    vv[0] = (1.0f - z) * vv[0] + z * vv[4];

                    if ((vv[0] < valMin) || (vv[0] > valMax)) r = 0.0f;

                    this->vertData.AsAt<float>(cnt)[0] = inVertData[0];
                    this->vertData.AsAt<float>(cnt)[1] = inVertData[1];
                    this->vertData.AsAt<float>(cnt)[2] = inVertData[2];
                    this->vertData.AsAt<float>(cnt)[3] = r;
                }
            }
            this->datahashOut++;
        }
    }

    outDpdc->SetDataHash(this->datahashOut);
    if (this->vertData.IsEmpty()) {
        // no particles for you :-/
        for (int i = outDpdc->GetParticleListCount() - 1; i >= 0; i--) {
            outDpdc->AccessParticles(i).SetVertexData(DirectionalParticleDataCall::Particles::VERTDATA_NONE, NULL);
        }

    } else {
        unsigned int cnt = outDpdc->GetParticleListCount();
        SIZE_T off = 0;
        for (unsigned int i = 0; i < cnt; i++) {
            outDpdc->AccessParticles(i).SetVertexData(DirectionalParticleDataCall::Particles::VERTDATA_FLOAT_XYZR, this->vertData.At(off));
            off += static_cast<SIZE_T>(outDpdc->AccessParticles(i).GetCount() * 4) * sizeof(float);
        }
    }

    return true;
}


/*
 * moldyn::DirPartFilter::getExtend
 */
bool moldyn::DirPartFilter::getExtend(Call& call) {
    DirectionalParticleDataCall *outDpdc = dynamic_cast<DirectionalParticleDataCall*>(&call);
    if (outDpdc == NULL) return false;
    DirectionalParticleDataCall *inDpdc = this->inParticlesDataSlot.CallAs<DirectionalParticleDataCall>();
    if (inDpdc == NULL) return false;

    *inDpdc = *outDpdc;
    if ((*inDpdc)(1)) {
        *outDpdc = *inDpdc;
        outDpdc->SetDataHash(this->datahashOut);
        return true;
    }

    return false;
}
