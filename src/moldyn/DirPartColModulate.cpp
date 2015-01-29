/*
 * DirPartColModulate.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "moldyn/DirPartColModulate.h"
#include "param/FloatParam.h"
#include "param/StringParam.h"
#include "vislib/graphics/ColourParser.h"
#include <climits>

using namespace megamol::core;


/*
 * moldyn::DirPartColModulate::DirPartColModulate
 */
moldyn::DirPartColModulate::DirPartColModulate(void) : Module(),
        inParticlesDataSlot("inPartData", "Input for oriented particle data"),
        inVolumeDataSlot("inVolData", "Input for volume data"),
        outDataSlot("outData", "Output of oriented particle data"),
        attributeSlot("attr", "The volume attribute to use"),
        attrMinValSlot("attrMinVal", "The minimum value"),
        attrMaxValSlot("attrMaxVal", "The maximum value"),
        baseColourSlot("col", "The base colour"),
        datahashOut(0), datahashParticlesIn(0), datahashVolumeIn(0),
        frameID(0), colData() {

    this->inParticlesDataSlot.SetCompatibleCall<moldyn::DirectionalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inParticlesDataSlot);

    this->inVolumeDataSlot.SetCompatibleCall<CallVolumeDataDescription>();
    this->MakeSlotAvailable(&this->inVolumeDataSlot);

    this->outDataSlot.SetCallback(moldyn::DirectionalParticleDataCall::ClassName(),
        moldyn::DirectionalParticleDataCall::FunctionName(0), &DirPartColModulate::getData);
    this->outDataSlot.SetCallback(moldyn::DirectionalParticleDataCall::ClassName(),
        moldyn::DirectionalParticleDataCall::FunctionName(1), &DirPartColModulate::getExtend);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->attributeSlot << new param::StringParam("0");
    this->MakeSlotAvailable(&this->attributeSlot);

    this->attrMinValSlot << new param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->attrMinValSlot);

    this->attrMaxValSlot << new param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->attrMaxValSlot);

    this->baseColourSlot << new param::StringParam("silver");
    this->MakeSlotAvailable(&this->baseColourSlot);
}


/*
 * moldyn::DirPartColModulate::~DirPartColModulate
 */
moldyn::DirPartColModulate::~DirPartColModulate(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * moldyn::DirPartColModulate::create
 */
bool moldyn::DirPartColModulate::create(void) {
    // intentionally empty
    return true;
}


/*
 * moldyn::DirPartColModulate::release
 */
void moldyn::DirPartColModulate::release(void) {
    // intentionally empty
}


/*
 * moldyn::DirPartColModulate::getData
 */
bool moldyn::DirPartColModulate::getData(Call& call) {
    bool rebuildColour = false;
    if (this->datahashOut == 0) rebuildColour = true;

    DirectionalParticleDataCall *outDpdc = dynamic_cast<DirectionalParticleDataCall*>(&call);
    if (outDpdc == NULL) return false;
    DirectionalParticleDataCall *inDpdc = this->inParticlesDataSlot.CallAs<DirectionalParticleDataCall>();
    if (inDpdc == NULL) return false;
    CallVolumeData *inCvd = this->inVolumeDataSlot.CallAs<CallVolumeData>();
    if (inCvd == NULL) return false;

    if (this->attributeSlot.IsDirty()) {
        rebuildColour = true;
        this->attributeSlot.ResetDirty();
    }
    if (this->attrMinValSlot.IsDirty()) {
        rebuildColour = true;
        this->attrMinValSlot.ResetDirty();
    }
    if (this->attrMaxValSlot.IsDirty()) {
        rebuildColour = true;
        this->attrMaxValSlot.ResetDirty();
    }
    if (this->baseColourSlot.IsDirty()) {
        rebuildColour = true;
        this->baseColourSlot.ResetDirty();
    }

    *inDpdc = *outDpdc;
    if (!(*inDpdc)(0)) {
        return false;
    }
    *outDpdc = *inDpdc;
    for (int i = outDpdc->GetParticleListCount() - 1; i >= 0; i--) {
        //switch (outDpdc->AccessParticles(i).GetColourDataType()) {
        //    case DirectionalParticleDataCall::Particles::COLDATA_FLOAT_I: // falls through
        //    case DirectionalParticleDataCall::Particles::COLDATA_NONE: // falls through
        //        return false;
        //}
        switch (outDpdc->AccessParticles(i).GetVertexDataType()) {
            case DirectionalParticleDataCall::Particles::VERTDATA_NONE: // falls through
            case DirectionalParticleDataCall::Particles::VERTDATA_SHORT_XYZ: // falls through
                return false;
        }
    }
    if (inDpdc->DataHash() != this->datahashParticlesIn) {
        rebuildColour = true;
        this->datahashParticlesIn = inDpdc->DataHash();
    }
    if (outDpdc->FrameID() != this->frameID) {
        rebuildColour = true;
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
        rebuildColour = true;
        this->datahashVolumeIn = inCvd->DataHash();
    }

    if (rebuildColour) {
        float bcR, bcG, bcB, r, g, b, v;
        vislib::graphics::ColourParser::FromString(this->baseColourSlot.Param<param::StringParam>()->Value(), bcR, bcG, bcB);
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
            float valRng = this->attrMaxValSlot.Param<param::FloatParam>()->Value() - valMin;
            if (vislib::math::IsEqual(valRng, 0.0f)) valRng = 1.0f;
            SIZE_T cnt = 0;
            unsigned int plCnt = outDpdc->GetParticleListCount();
            for (unsigned int i = 0; i < plCnt; i++) {
                cnt += static_cast<SIZE_T>(outDpdc->AccessParticles(i).GetCount());
            }
            this->colData.EnforceSize(cnt * 3);
            cnt = 0;
            for (unsigned int i = 0; i < plCnt; i++) {
                SIZE_T pCnt = static_cast<SIZE_T>(outDpdc->AccessParticles(i).GetCount());
                const unsigned char *inColDat = static_cast<const unsigned char*>(outDpdc->AccessParticles(i).GetColourData());
                unsigned int inColStep = outDpdc->AccessParticles(i).GetColourDataStride();
                bool isFloat = false;
                switch (outDpdc->AccessParticles(i).GetColourDataType()) {
                case DirectionalParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                    inColStep = vislib::math::Max<unsigned int>(inColStep, 3 * sizeof(float));
                    isFloat = true;
                    break;
                case DirectionalParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                    inColStep = vislib::math::Max<unsigned int>(inColStep, 4 * sizeof(float));
                    isFloat = true;
                    break;
                case DirectionalParticleDataCall::Particles::COLDATA_UINT8_RGB:
                    inColStep = vislib::math::Max<unsigned int>(inColStep, 3 * sizeof(unsigned char));
                    break;
                case DirectionalParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                    inColStep = vislib::math::Max<unsigned int>(inColStep, 4 * sizeof(unsigned char));
                    break;
                }
                const unsigned char *inPosDat = static_cast<const unsigned char*>(outDpdc->AccessParticles(i).GetVertexData());
                unsigned int inPosStep = outDpdc->AccessParticles(i).GetVertexDataStride();
                switch (outDpdc->AccessParticles(i).GetVertexDataType()) {
                case DirectionalParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    inPosStep = vislib::math::Max<unsigned int>(inPosStep, 3 * sizeof(float));
                    break;
                case DirectionalParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    inPosStep = vislib::math::Max<unsigned int>(inPosStep, 4 * sizeof(float));
                    break;
                }

                if (outDpdc->AccessParticles(i).GetColourDataType() == DirectionalParticleDataCall::Particles::COLDATA_FLOAT_I
                    || outDpdc->AccessParticles(i).GetColourDataType() == DirectionalParticleDataCall::Particles::COLDATA_NONE) {
                        continue;
                }
                for (SIZE_T p = 0; p < pCnt; p++, cnt += 3, inColDat += inColStep, inPosDat += inPosStep) {
                    float x = (reinterpret_cast<const float*>(inPosDat)[0] - volBB.Left()) / volBB.Width();
                    float y = (reinterpret_cast<const float*>(inPosDat)[1] - volBB.Bottom()) / volBB.Height();
                    float z = (reinterpret_cast<const float*>(inPosDat)[2] - volBB.Back()) / volBB.Depth();
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

                    v = (1.0f - z) * vv[0] + z * vv[4];
                    v = (v - valMin) / valRng;

                    v = (3.0f - 2.0f * v) * v * v;

                    if (isFloat) {
                        r = reinterpret_cast<const float*>(inColDat)[0];
                        g = reinterpret_cast<const float*>(inColDat)[1];
                        b = reinterpret_cast<const float*>(inColDat)[2];
                    } else {
                        r = inColDat[0];
                        g = inColDat[1];
                        b = inColDat[2];
                    }

                    r = v * r + (1.0f - v) * bcR;
                    g = v * g + (1.0f - v) * bcG;
                    b = v * b + (1.0f - v) * bcB;

                    this->colData.AsAt<unsigned char>(cnt)[0] = static_cast<unsigned char>(vislib::math::Clamp(r, 0.0f, 1.0f) * 255.0f);
                    this->colData.AsAt<unsigned char>(cnt)[1] = static_cast<unsigned char>(vislib::math::Clamp(g, 0.0f, 1.0f) * 255.0f);
                    this->colData.AsAt<unsigned char>(cnt)[2] = static_cast<unsigned char>(vislib::math::Clamp(b, 0.0f, 1.0f) * 255.0f);
                }
            }
            this->datahashOut++;
        }
    }

    outDpdc->SetDataHash(this->datahashOut);
    if (this->colData.IsEmpty()) {
        // no colour for you :-/
        for (int i = outDpdc->GetParticleListCount() - 1; i >= 0; i--) {
            outDpdc->AccessParticles(i).SetColourData(DirectionalParticleDataCall::Particles::COLDATA_NONE, NULL);
        }

    } else {
        unsigned int cnt = outDpdc->GetParticleListCount();
        SIZE_T off = 0;
        for (unsigned int i = 0; i < cnt; i++) {

            switch (outDpdc->AccessParticles(i).GetColourDataType()) {
                case DirectionalParticleDataCall::Particles::COLDATA_FLOAT_I: // falls through
                case DirectionalParticleDataCall::Particles::COLDATA_NONE: // falls through
                    //return false;
                    outDpdc->AccessParticles(i).SetColourData(DirectionalParticleDataCall::Particles::COLDATA_NONE, NULL);
                    break;
                default:
                    outDpdc->AccessParticles(i).SetColourData(DirectionalParticleDataCall::Particles::COLDATA_UINT8_RGB, this->colData.At(off));
                    off += static_cast<SIZE_T>(outDpdc->AccessParticles(i).GetCount() * 3);
            }

        }
    }

    return true;
}


/*
 * moldyn::DirPartColModulate::getExtend
 */
bool moldyn::DirPartColModulate::getExtend(Call& call) {
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
