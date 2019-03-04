/*
 * AddParticleColours.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/moldyn/AddParticleColours.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ButtonParam.h"
#include "vislib/RawStorage.h"
#include "vislib/RawStorageWriter.h"
#include <algorithm>

using namespace megamol::core;

/****************************************************************************/

/*
 * moldyn::AddParticleColours::Unlocker::Unlocker
 */
moldyn::AddParticleColours::Unlocker::Unlocker(
        MultiParticleDataCall::Unlocker *inner)
        : MultiParticleDataCall::Unlocker(), inner(inner) {
    // intentionally empty ATM
}


/*
 * moldyn::AddParticleColours::Unlocker::~Unlocker
 */
moldyn::AddParticleColours::Unlocker::~Unlocker(void) {
    this->Unlock();
}


/*
 * moldyn::AddParticleColours::Unlocker::Unlock
 */
void moldyn::AddParticleColours::Unlocker::Unlock(void) {
    if (this->inner != NULL) {
        this->inner->Unlock();
        SAFE_DELETE(this->inner);
    }
}


/****************************************************************************/


/*
 * moldyn::AddParticleColours::AddParticleColours
 */
moldyn::AddParticleColours::AddParticleColours(void) : Module(),
        putDataSlot("putdata", "Connects from the data consumer"),
        getDataSlot("getdata", "Connects to the data source"),
        getTFSlot("gettransferfunction", "Connects to the transfer function module"),
        rebuildButtonSlot("rebuild", "Forces rebuild of colour data"),
        lastFrame(0), colData(),
        colFormat(view::CallGetTransferFunction::TEXTURE_FORMAT_RGB),
        updateHash() {

    this->putDataSlot.SetCallback("MultiParticleDataCall", "GetData",
        &AddParticleColours::getDataCallback);
    this->putDataSlot.SetCallback("MultiParticleDataCall", "GetExtent",
        &AddParticleColours::getExtentCallback);
    this->MakeSlotAvailable(&this->putDataSlot);

    this->getDataSlot.SetCompatibleCall<moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->rebuildButtonSlot << new param::ButtonParam();
    this->MakeSlotAvailable(&this->rebuildButtonSlot);
}


/*
 * moldyn::AddParticleColours::~AddParticleColours
 */
moldyn::AddParticleColours::~AddParticleColours(void) {
    this->Release();
}


/*
 * moldyn::AddParticleColours::create
 */
bool moldyn::AddParticleColours::create(void) {
    return true;
}


/*
 * moldyn::AddParticleColours::release
 */
void moldyn::AddParticleColours::release(void) {
    // intentionally empty ATM
}


/*
 * moldyn::AddParticleColours::getDataCallback
 */
bool moldyn::AddParticleColours::getDataCallback(Call& caller) {
    static vislib::RawStorage updateHash;
    static vislib::RawStorageWriter uhWriter(updateHash);
    MultiParticleDataCall *inCall = dynamic_cast<MultiParticleDataCall*>(&caller);
    if (inCall == NULL) return false;

    MultiParticleDataCall *outCall = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if (outCall == NULL) return false;

    *outCall = *inCall;

    if ((*outCall)(0)) {
        uhWriter.SetPosition(0);

        SIZE_T cntCol = 0;
        for (unsigned int i = 0; i < outCall->GetParticleListCount(); i++) {
            MultiParticleDataCall::Particles &part = outCall->AccessParticles(i);
            if (part.GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
                SIZE_T cnt = static_cast<SIZE_T>(part.GetCount());
                cntCol += cnt;
                uhWriter.Write(part.GetMinColourIndexValue());
                uhWriter.Write(part.GetMaxColourIndexValue());

                const unsigned char *cd = static_cast<const unsigned char*>(part.GetColourData());
                unsigned int stride = std::max<unsigned int>(part.GetColourDataStride(), sizeof(float));
                unsigned int i2 = static_cast<unsigned int>(cnt - 1) / 2;
                unsigned int i3 = static_cast<unsigned int>(cnt - 1);
                uhWriter.Write(*reinterpret_cast<const float*>(cd + (0  * stride)));
                uhWriter.Write(*reinterpret_cast<const float*>(cd + (i2 * stride)));
                uhWriter.Write(*reinterpret_cast<const float*>(cd + (i3 * stride)));
            }
        }

        if (cntCol > 0) {
            int colcompcnt = (this->colFormat == view::CallGetTransferFunction::TEXTURE_FORMAT_RGB) ? 3 : 4;

            if ((this->lastFrame != outCall->FrameID()) || ((cntCol * colcompcnt) != this->colData.GetSize())
                    || this->rebuildButtonSlot.IsDirty() || (uhWriter.End() != this->updateHash.GetSize())
                    || (::memcmp(updateHash.As<void>(), this->updateHash.As<void>(), uhWriter.End()) != 0)) {

                this->updateHash = updateHash;
                this->updateHash.EnforceSize(uhWriter.End(), true);
                this->rebuildButtonSlot.ResetDirty();

                // build colour data
                this->lastFrame = outCall->FrameID();

                vislib::RawStorage texDat;

                view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
                if ((cgtf != NULL) && ((*cgtf)(0))) {
                    this->colFormat = cgtf->OpenGLTextureFormat();
                    colcompcnt = (this->colFormat == view::CallGetTransferFunction::TEXTURE_FORMAT_RGB) ? 3 : 4;

                    ::glGetError();
                    ::glEnable(GL_TEXTURE_1D);
                    ::glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());

                    int texSize = 0;
                    ::glGetTexLevelParameteriv(GL_TEXTURE_1D, 0, GL_TEXTURE_WIDTH, &texSize);

                    if (::glGetError() == GL_NO_ERROR) {
                        texDat.EnforceSize(texSize * 4 * colcompcnt);
                        ::glGetTexImage(GL_TEXTURE_1D, 0, this->colFormat, GL_FLOAT, texDat.As<void>());
                        if (::glGetError() != GL_NO_ERROR) {
                            texDat.EnforceSize(0);
                        }
                    }

                    ::glBindTexture(GL_TEXTURE_1D, 0);
                    ::glDisable(GL_TEXTURE_1D);
                }

                this->colData.EnforceSize(cntCol * colcompcnt);

                unsigned int texDatSize = 2;
                if (texDat.GetSize() < static_cast<SIZE_T>(2 * 4 * colcompcnt)) {
                    texDat.EnforceSize(2 * 4 * colcompcnt);
                    *texDat.AsAt<float>(0) = 0.0f;
                    *texDat.AsAt<float>(4) = 0.0f;
                    *texDat.AsAt<float>(8) = 0.0f;
                    *texDat.AsAt<float>(12) = 1.0f;
                    *texDat.AsAt<float>(16) = 1.0f;
                    *texDat.AsAt<float>(20) = 1.0f;
                    if (colcompcnt == 4) {
                        *texDat.AsAt<float>(24) = 1.0f;
                        *texDat.AsAt<float>(28) = 1.0f;
                    }
                } else {
                    texDatSize = static_cast<unsigned int>(texDat.GetSize() / (4 * colcompcnt));
                }
                texDatSize--;

                cntCol = 0;
                for (unsigned int i = 0; i < outCall->GetParticleListCount(); i++) {
                    MultiParticleDataCall::Particles &parts = outCall->AccessParticles(i);
                    if (parts.GetColourDataType() != MultiParticleDataCall::Particles::COLDATA_FLOAT_I) continue;
                    const float *values = static_cast<const float*>(parts.GetColourData());
                    unsigned int stride = std::max<unsigned int>(parts.GetColourDataStride(), sizeof(float));


                    for (UINT64 j = 0; j < parts.GetCount(); j++) {
                        float v = (*values - parts.GetMinColourIndexValue()) / (parts.GetMaxColourIndexValue() - parts.GetMinColourIndexValue());
                        values = reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(values) + stride);
                        if (v < 0.0f) v = 0.0f;
                        else if (v > 1.0f) v = 1.0f;

                        v *= static_cast<float>(texDatSize);

                        int idx = static_cast<int>(v);
                        v -= static_cast<float>(idx);
                        float w = 1.0f - v;
                        if (idx == static_cast<int>(texDatSize)) {
                            idx--;
                            v = 1.0f;
                            w = 0.0;
                        }

                        unsigned int stride = sizeof(float) *colcompcnt;
                        *this->colData.AsAt<BYTE>(cntCol++) = static_cast<BYTE>(255.0f * (
                            *texDat.AsAt<float>(idx * stride) * w + *texDat.AsAt<float>((idx + 1) * stride) * v));
                        *this->colData.AsAt<BYTE>(cntCol++) = static_cast<BYTE>(255.0f * (
                            *texDat.AsAt<float>(idx * stride + 4) * w + *texDat.AsAt<float>((idx + 1) * stride + 4) * v));
                        *this->colData.AsAt<BYTE>(cntCol++) = static_cast<BYTE>(255.0f * (
                            *texDat.AsAt<float>(idx * stride + 8) * w + *texDat.AsAt<float>((idx + 1) * stride + 8) * v));
                        if (colcompcnt == 4) {
                            *this->colData.AsAt<BYTE>(cntCol++) = static_cast<BYTE>(255.0f * (
                                *texDat.AsAt<float>(idx * stride + 12) * w + *texDat.AsAt<float>((idx + 1) * stride + 12) * v));
                        }
                    }
                }

            }

            *inCall = *outCall;
            outCall->SetUnlocker(NULL, false);
            inCall->SetUnlocker(new Unlocker(inCall->GetUnlocker()), false);

            cntCol = 0;
            for (unsigned int i = 0; i < inCall->GetParticleListCount(); i++) {
                if (inCall->AccessParticles(i).GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
                    inCall->AccessParticles(i).SetColourData(
                        (colcompcnt == 3)
                        ? MultiParticleDataCall::Particles::COLDATA_UINT8_RGB
                        : MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA,
                        this->colData.At(cntCol));
                    cntCol += colcompcnt * static_cast<SIZE_T>(inCall->AccessParticles(i).GetCount());
                }
            }

            return true;

        } else {
            *inCall = *outCall;
            outCall->SetUnlocker(NULL, false);
            return true;
        }
    }

    return false;
}


/*
 * moldyn::AddParticleColours::getExtentCallback
 */
bool moldyn::AddParticleColours::getExtentCallback(Call& caller) {
    MultiParticleDataCall *inCall = dynamic_cast<MultiParticleDataCall*>(&caller);
    if (inCall == NULL) return false;

    MultiParticleDataCall *outCall = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if (outCall == NULL) return false;

    *outCall = *inCall;

    if ((*outCall)(1)) {
        outCall->SetUnlocker(NULL, false);
        *inCall = *outCall;
        return true;
    }

    return false;
}
