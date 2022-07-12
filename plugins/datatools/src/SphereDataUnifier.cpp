/*
 * SphereDataUnifier.cpp
 *
 * Copyright (C) 2012 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#include "SphereDataUnifier.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "vislib/RawStorage.h"

using namespace megamol;


/*
 * datatools::SphereDataUnifier::SphereDataUnifier
 */
datatools::SphereDataUnifier::SphereDataUnifier(void)
        : Module()
        , putDataSlot("putdata", "Connects from the data consumer")
        , getDataSlot("getdata", "Connects to the data source")
        , inDataHash(0)
        , outDataHash(0)
        , data()
        , bbox()
        , cbox() {

    this->putDataSlot.SetCallback("MultiParticleDataCall", "GetData", &SphereDataUnifier::getDataCallback);
    this->putDataSlot.SetCallback("MultiParticleDataCall", "GetExtent", &SphereDataUnifier::getExtentCallback);
    this->MakeSlotAvailable(&this->putDataSlot);

    this->getDataSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);
}


/*
 * datatools::SphereDataUnifier::~SphereDataUnifier
 */
datatools::SphereDataUnifier::~SphereDataUnifier(void) {
    this->Release();
}


/*
 * datatools::SphereDataUnifier::create
 */
bool datatools::SphereDataUnifier::create(void) {
    return true;
}


/*
 * datatools::SphereDataUnifier::release
 */
void datatools::SphereDataUnifier::release(void) {
    // intentionally empty
}


/*
 * datatools::SphereDataUnifier::getDataCallback
 */
bool datatools::SphereDataUnifier::getDataCallback(core::Call& caller) {
    using geocalls::MultiParticleDataCall;
    MultiParticleDataCall* inCall = dynamic_cast<MultiParticleDataCall*>(&caller);
    if (inCall == NULL)
        return false;

    MultiParticleDataCall* outCall = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if (outCall == NULL)
        return false;

    *outCall = *inCall;

    if ((*outCall)(0)) {
        unsigned int listCnt = outCall->GetParticleListCount();
        SIZE_T datPos = 0;

        if (outCall->DataHash() != this->inDataHash) {
            this->inDataHash = outCall->DataHash();
            this->outDataHash++;

            this->bbox.Set(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            this->cbox.Set(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            bool first = true;

            for (unsigned int li = 0; li < listCnt; li++) {
                MultiParticleDataCall::Particles& outP = outCall->AccessParticles(li);
                UINT64 pCnt = outP.GetCount();

                SIZE_T deS = 0;
                MultiParticleDataCall::Particles::VertexDataType vdt = outP.GetVertexDataType();
                switch (vdt) {
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    datPos += sizeof(float);
                    deS = sizeof(float) * 3;
                    {
                        const char* vD = static_cast<const char*>(outP.GetVertexData());
                        SIZE_T vDs = vislib::math::Max<SIZE_T>(3 * sizeof(float), outP.GetVertexDataStride());
                        for (UINT64 pi = 0; pi < pCnt; pi++, vD += vDs) {
                            const float* vDf = reinterpret_cast<const float*>(vD);
                            this->accumExt(first, vDf[0], vDf[1], vDf[2], outP.GetGlobalRadius());
                        }
                    }
                    break;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    deS = sizeof(float) * 4;
                    {
                        const char* vD = static_cast<const char*>(outP.GetVertexData());
                        SIZE_T vDs = vislib::math::Max<SIZE_T>(4 * sizeof(float), outP.GetVertexDataStride());
                        for (UINT64 pi = 0; pi < pCnt; pi++, vD += vDs) {
                            const float* vDf = reinterpret_cast<const float*>(vD);
                            this->accumExt(first, vDf[0], vDf[1], vDf[2], vDf[3]);
                        }
                    }
                    break;
                case MultiParticleDataCall::Particles::VERTDATA_NONE:
                    deS = 0;
                    break;
                case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
                    datPos += sizeof(float);
                    deS = sizeof(float) * 3; // is most probably bullshit anyway
                    {
                        const char* vD = static_cast<const char*>(outP.GetVertexData());
                        SIZE_T vDs = vislib::math::Max<SIZE_T>(3 * sizeof(float), outP.GetVertexDataStride());
                        for (UINT64 pi = 0; pi < pCnt; pi++, vD += vDs) {
                            const signed short* vDf = reinterpret_cast<const signed short*>(vD);
                            this->accumExt(first, static_cast<float>(vDf[0]), static_cast<float>(vDf[1]),
                                static_cast<float>(vDf[2]), outP.GetGlobalRadius());
                        }
                    }
                    break;
                default:
                    deS = 0;
                    break;
                }

                datPos += static_cast<SIZE_T>(deS * pCnt);
            }

            this->data.AssertSize(datPos);
            datPos = 0;

            float scale = 1.0f;
            float xOff = 0.0f;
            float yOff = 0.0f;
            float zOff = 0.0f;

            if (!this->bbox.IsEmpty()) {
                vislib::math::Point<float, 3> bbcp = this->bbox.CalcCenter();
                xOff = -bbcp.X();
                yOff = -bbcp.Y();
                zOff = -bbcp.Z();
                scale = 1.0f / this->bbox.LongestEdge();
            }

            this->bbox.Move(xOff, yOff, zOff);
            this->bbox *= scale;
            this->cbox.Move(xOff, yOff, zOff);
            this->cbox *= scale;

            for (unsigned int li = 0; li < listCnt; li++) {
                MultiParticleDataCall::Particles& outP = outCall->AccessParticles(li);
                UINT64 pCnt = outP.GetCount();

                SIZE_T deS = 0;
                MultiParticleDataCall::Particles::VertexDataType vdt = outP.GetVertexDataType();
                switch (vdt) {
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    *this->data.AsAt<float>(datPos) = outP.GetGlobalRadius() * scale;
                    datPos += sizeof(float);
                    deS = sizeof(float) * 3;
                    {
                        const char* vD = static_cast<const char*>(outP.GetVertexData());
                        SIZE_T vDs = vislib::math::Max<SIZE_T>(3 * sizeof(float), outP.GetVertexDataStride());
                        for (UINT64 pi = 0; pi < pCnt; pi++, vD += vDs) {
                            const float* vDf = reinterpret_cast<const float*>(vD);
                            *this->data.AsAt<float>(static_cast<SIZE_T>(datPos + pi * deS + 0 * sizeof(float))) =
                                (vDf[0] + xOff) * scale;
                            *this->data.AsAt<float>(static_cast<SIZE_T>(datPos + pi * deS + 1 * sizeof(float))) =
                                (vDf[1] + yOff) * scale;
                            *this->data.AsAt<float>(static_cast<SIZE_T>(datPos + pi * deS + 2 * sizeof(float))) =
                                (vDf[2] + zOff) * scale;
                        }
                    }
                    break;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    deS = sizeof(float) * 4;
                    {
                        const char* vD = static_cast<const char*>(outP.GetVertexData());
                        SIZE_T vDs = vislib::math::Max<SIZE_T>(4 * sizeof(float), outP.GetVertexDataStride());
                        for (UINT64 pi = 0; pi < pCnt; pi++, vD += vDs) {
                            const float* vDf = reinterpret_cast<const float*>(vD);
                            *this->data.AsAt<float>(static_cast<SIZE_T>(datPos + pi * deS + 0 * sizeof(float))) =
                                (vDf[0] + xOff) * scale;
                            *this->data.AsAt<float>(static_cast<SIZE_T>(datPos + pi * deS + 1 * sizeof(float))) =
                                (vDf[1] + yOff) * scale;
                            *this->data.AsAt<float>(static_cast<SIZE_T>(datPos + pi * deS + 2 * sizeof(float))) =
                                (vDf[2] + zOff) * scale;
                            *this->data.AsAt<float>(static_cast<SIZE_T>(datPos + pi * deS + 3 * sizeof(float))) =
                                vDf[3] * scale;
                        }
                    }
                    break;
                case MultiParticleDataCall::Particles::VERTDATA_NONE:
                    deS = 0;
                    break;
                case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
                    *this->data.AsAt<float>(datPos) = outP.GetGlobalRadius() * scale;
                    datPos += sizeof(float);
                    deS = sizeof(float) * 3; // is most probably bullshit anyway
                    {
                        const char* vD = static_cast<const char*>(outP.GetVertexData());
                        SIZE_T vDs = vislib::math::Max<SIZE_T>(3 * sizeof(float), outP.GetVertexDataStride());
                        for (UINT64 pi = 0; pi < pCnt; pi++, vD += vDs) {
                            const signed short* vDf = reinterpret_cast<const signed short*>(vD);
                            *this->data.AsAt<float>(static_cast<SIZE_T>(datPos + pi * deS + 0 * sizeof(float))) =
                                (static_cast<float>(vDf[0]) + xOff) * scale;
                            *this->data.AsAt<float>(static_cast<SIZE_T>(datPos + pi * deS + 1 * sizeof(float))) =
                                (static_cast<float>(vDf[1]) + yOff) * scale;
                            *this->data.AsAt<float>(static_cast<SIZE_T>(datPos + pi * deS + 2 * sizeof(float))) =
                                (static_cast<float>(vDf[2]) + zOff) * scale;
                        }
                    }
                    break;
                default:
                    deS = 0;
                    break;
                }

                datPos += static_cast<SIZE_T>(deS * pCnt);
            }

            ASSERT(datPos == this->data.GetSize());

            datPos = 0;
        }

        inCall->SetDataHash(this->outDataHash);
        inCall->AccessBoundingBoxes().Clear();
        inCall->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
        inCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->cbox);
        inCall->SetParticleListCount(listCnt);
        inCall->SetFrameID(outCall->FrameID());
        inCall->SetUnlocker(outCall->GetUnlocker());
        outCall->SetUnlocker(NULL, false);

        for (unsigned int li = 0; li < listCnt; li++) {
            MultiParticleDataCall::Particles& outP = outCall->AccessParticles(li);
            MultiParticleDataCall::Particles& inP = inCall->AccessParticles(li);

            UINT64 pCnt = outP.GetCount();
            inP.SetCount(pCnt);
            inP.SetColourData(outP.GetColourDataType(), outP.GetColourData(), outP.GetColourDataStride());
            inP.SetColourMapIndexValues(outP.GetMinColourIndexValue(), outP.GetMaxColourIndexValue());
            inP.SetGlobalColour(outP.GetGlobalColour()[0], outP.GetGlobalColour()[1], outP.GetGlobalColour()[2],
                outP.GetGlobalColour()[3]);

            SIZE_T deS = 0;
            MultiParticleDataCall::Particles::VertexDataType vdt = outP.GetVertexDataType();
            switch (vdt) {
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                inP.SetGlobalRadius(*this->data.AsAt<float>(datPos));
                datPos += sizeof(float);
                deS = sizeof(float) * 3;
                break;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                inP.SetGlobalRadius(0.0f);
                deS = sizeof(float) * 4;
                break;
            case MultiParticleDataCall::Particles::VERTDATA_NONE:
                inP.SetGlobalRadius(0.0f);
                deS = 0;
                break;
            case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
                inP.SetGlobalRadius(*this->data.AsAt<float>(datPos));
                datPos += sizeof(float);
                deS = sizeof(float) * 3; // is most probably bullshit anyway
                break;
            default:
                inP.SetGlobalRadius(0.0f);
                deS = 0;
                break;
            }

            if (pCnt * deS == 0) {
                inP.SetVertexData(MultiParticleDataCall::Particles::VERTDATA_NONE, NULL, 0);
            } else {
                inP.SetVertexData((deS == sizeof(float) * 3) ? MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ
                                                             : MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR,
                    this->data.AsAt<float>(datPos), 0);
                datPos += static_cast<SIZE_T>(deS * pCnt);
            }
        }

        return true;
    }

    return false;
}


/*
 * datatools::SphereDataUnifier::getExtentCallback
 */
bool datatools::SphereDataUnifier::getExtentCallback(core::Call& caller) {
    using geocalls::MultiParticleDataCall;
    MultiParticleDataCall* inCall = dynamic_cast<MultiParticleDataCall*>(&caller);
    if (inCall == NULL)
        return false;

    MultiParticleDataCall* outCall = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if (outCall == NULL)
        return false;

    *outCall = *inCall;

    if ((*outCall)(1)) {
        outCall->SetUnlocker(NULL, false);
        *inCall = *outCall;

        this->getDataCallback(caller); // this is ... ... at least working

        inCall->AccessBoundingBoxes().Clear();
        inCall->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
        inCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->cbox);

        return true;
    }

    return false;
}


/*
 * datatools::SphereDataUnifier::accumExt
 */
void datatools::SphereDataUnifier::accumExt(bool& first, float x, float y, float z, float r) {
    vislib::math::Cuboid<float> b(x - r, y - r, z - r, x + r, y + r, z + r);
    vislib::math::Point<float, 3> p(x, y, z);

    if (first) {
        first = false;
        this->bbox.Set(p.X(), p.Y(), p.Z(), p.X(), p.Y(), p.Z());
        this->cbox = b;
    } else {
        this->bbox.GrowToPoint(p);
        this->cbox.Union(b);
    }
}
