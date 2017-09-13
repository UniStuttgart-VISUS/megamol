/*
 * ParticleListMergeModule.cpp
 *
 * Copyright (C) 2014 by CGV TU Dresden
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ParticleListMergeModule.h"
#include <algorithm>

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::ParticleListMergeModule::ParticleListMergeModule
 */
datatools::ParticleListMergeModule::ParticleListMergeModule(void)
        : AbstractParticleManipulator("outData", "inData"), tfq(),
        dataHash(0), frameId(0), parts(), data() {
    this->MakeSlotAvailable(this->tfq.GetSlot());
}


/*
 * datatools::ParticleListMergeModule::~ParticleListMergeModule
 */
datatools::ParticleListMergeModule::~ParticleListMergeModule(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * datatools::ParticleListMergeModule::manipulateData
 */
bool datatools::ParticleListMergeModule::manipulateData(
        megamol::core::moldyn::MultiParticleDataCall& outData,
        megamol::core::moldyn::MultiParticleDataCall& inData) {

    if ((this->frameId != inData.FrameID()) || (this->dataHash != inData.DataHash()) || (inData.DataHash() == 0)) {
        this->frameId = inData.FrameID();
        this->dataHash = inData.DataHash();
        this->setData(inData);
    }
    inData.Unlock();

    outData.SetDataHash(this->dataHash);
    outData.SetFrameID(this->frameId);
    outData.SetParticleListCount(1);
    outData.AccessParticles(0) = this->parts;
    outData.SetUnlocker(nullptr); // HAZARD: we could have one ...

    return true;
}


/*
 * datatools::ParticleListMergeModule::setData
 */
void datatools::ParticleListMergeModule::setData(core::moldyn::MultiParticleDataCall& inDat) {
    using core::moldyn::SimpleSphericalParticles;

    // analyze lists
    bool first = true;

    SimpleSphericalParticles::VertexDataType vdType = SimpleSphericalParticles::VERTDATA_NONE;
    SimpleSphericalParticles::ColourDataType cdType = SimpleSphericalParticles::COLDATA_NONE;
    uint8_t gc[4];
    float gr = 0.0f;
    float gcimin = 0.0f;
    float gcimax = 0.0f;

    uint64_t partCnt = 0;
    for (unsigned int li = 0; li < inDat.GetParticleListCount(); li++) {
        core::moldyn::MultiParticleDataCall::Particles& p = inDat.AccessParticles(li);
        if (p.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_NONE) continue;
        if (p.GetCount() == 0) continue;

        if (first) {
            first = false;

            vdType = p.GetVertexDataType();
            cdType = p.GetColourDataType();
            ::memcpy(gc, p.GetGlobalColour(), 4);
            gcimin = p.GetMinColourIndexValue();
            gcimax = p.GetMaxColourIndexValue();
            gr = p.GetGlobalRadius();

        } else {
            const float my_eps = 0.0001f;
            // vertex data type
            if (vdType != SimpleSphericalParticles::VERTDATA_FLOAT_XYZR) {
                if (vdType != p.GetVertexDataType()) {
                    if (p.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZR) {
                        vdType = SimpleSphericalParticles::VERTDATA_FLOAT_XYZR;
                    }
                } else if ((vdType == SimpleSphericalParticles::VERTDATA_FLOAT_XYZ)
                        || (vdType == SimpleSphericalParticles::VERTDATA_SHORT_XYZ)) {
                    if (!vislib::math::IsEqual(gr, p.GetGlobalRadius(), my_eps)) {
                        gr = 0.0f;
                        vdType = SimpleSphericalParticles::VERTDATA_FLOAT_XYZR;
                    }
                }
            }

            // color data type
            if (cdType != SimpleSphericalParticles::COLDATA_FLOAT_RGBA) {
                if (cdType != p.GetColourDataType()) {
                    switch (p.GetColourDataType()) {
                    case SimpleSphericalParticles::COLDATA_UINT8_RGB:
                        switch (cdType) {
                        //case SimpleSphericalParticles::COLDATA_UINT8_RGB: break; // cannot happen
                        case SimpleSphericalParticles::COLDATA_NONE: cdType = SimpleSphericalParticles::COLDATA_UINT8_RGB; break;
                        case SimpleSphericalParticles::COLDATA_UINT8_RGBA: cdType = SimpleSphericalParticles::COLDATA_UINT8_RGBA; break;
                        case SimpleSphericalParticles::COLDATA_FLOAT_RGB: cdType = SimpleSphericalParticles::COLDATA_FLOAT_RGB; break;
                        case SimpleSphericalParticles::COLDATA_FLOAT_I: // fall through
                        case SimpleSphericalParticles::COLDATA_FLOAT_RGBA: cdType = SimpleSphericalParticles::COLDATA_FLOAT_RGBA; break;
                        }
                        break;
                    case SimpleSphericalParticles::COLDATA_NONE: //< use global colour
                        // global color is UINT8_RGBA
                        // falls through
                    case SimpleSphericalParticles::COLDATA_UINT8_RGBA:
                        switch (cdType) {
                        case SimpleSphericalParticles::COLDATA_NONE: // fall through
                        case SimpleSphericalParticles::COLDATA_UINT8_RGB: cdType = SimpleSphericalParticles::COLDATA_UINT8_RGBA; break;
                        //case SimpleSphericalParticles::COLDATA_UINT8_RGBA: // cannot happen
                        case SimpleSphericalParticles::COLDATA_FLOAT_RGB: // fall through
                        case SimpleSphericalParticles::COLDATA_FLOAT_I: // fall through
                        case SimpleSphericalParticles::COLDATA_FLOAT_RGBA: cdType = SimpleSphericalParticles::COLDATA_FLOAT_RGBA; break;
                        }
                        break;
                    case SimpleSphericalParticles::COLDATA_FLOAT_RGB:
                        switch (cdType) {
                        case SimpleSphericalParticles::COLDATA_NONE: // fall through
                        case SimpleSphericalParticles::COLDATA_UINT8_RGB: cdType = SimpleSphericalParticles::COLDATA_FLOAT_RGB; break;
                        case SimpleSphericalParticles::COLDATA_UINT8_RGBA: // fall through
                        //case SimpleSphericalParticles::COLDATA_FLOAT_RGB: // cannot happen
                        case SimpleSphericalParticles::COLDATA_FLOAT_I: // fall through
                        case SimpleSphericalParticles::COLDATA_FLOAT_RGBA: cdType = SimpleSphericalParticles::COLDATA_FLOAT_RGBA; break;
                        }
                        break;
                    case SimpleSphericalParticles::COLDATA_FLOAT_I: //< single float value to be mapped by a transfer function
                        // transfer function color is FLOAT_RGBA
                        // falls through
                    case SimpleSphericalParticles::COLDATA_FLOAT_RGBA:
                        cdType = SimpleSphericalParticles::COLDATA_FLOAT_RGBA;
                        break;
                    }
                } else {
                    // cdType == p.GetColourDataType != COLDATA_FLOAT_RGBA
                    if (cdType == SimpleSphericalParticles::COLDATA_NONE) {
                        if (::memcmp(gc, p.GetGlobalColour(), 4) != 0) {
                            if ((gc[3] != 255) || (p.GetGlobalColour()[3] != 255)) {
                                cdType = SimpleSphericalParticles::COLDATA_UINT8_RGBA;
                            } else {
                                cdType = SimpleSphericalParticles::COLDATA_UINT8_RGB;
                            }
                        }
                    } else if (cdType == SimpleSphericalParticles::COLDATA_FLOAT_I) {
                        if (!vislib::math::IsEqual(gcimin, p.GetMinColourIndexValue(), my_eps)
                                || !vislib::math::IsEqual(gcimax, p.GetMaxColourIndexValue(), my_eps)) {
                            // I will be rescaled for each list
                            gcimin = 0.0f;
                            gcimax = 1.0f;
                        }
                    }
                }
            }
        }

        partCnt += p.GetCount();
    }

    // prepare local storage
    this->parts.SetCount(0);
    if (first) { // no data was found
        this->parts.SetVertexData(SimpleSphericalParticles::VERTDATA_NONE, nullptr);
        this->parts.SetColourData(SimpleSphericalParticles::COLDATA_NONE, nullptr);
        this->data.EnforceSize(0);
        return;
    }

    if (cdType != SimpleSphericalParticles::COLDATA_FLOAT_I) {
        gcimin = 0.0f;
        gcimax = 1.0f;
    }
    if (cdType != SimpleSphericalParticles::COLDATA_NONE) {
        gc[0] = gc[1] = gc[2] = 127;
        gc[3] = 255;
    }

    this->tfq.Clear();

    // finally copy data
    this->parts.SetCount(partCnt);
    this->parts.SetGlobalRadius(gr);
    this->parts.SetGlobalColour(gc[0], gc[1], gc[2], gc[3]);
    this->parts.SetColourMapIndexValues(gcimin, gcimax);

    unsigned int bpp = 0;
    switch (vdType) {
    case SimpleSphericalParticles::VERTDATA_NONE: throw std::exception();
    case SimpleSphericalParticles::VERTDATA_FLOAT_XYZ: bpp = 12; break;
    case SimpleSphericalParticles::VERTDATA_FLOAT_XYZR: bpp = 16; break;
    case SimpleSphericalParticles::VERTDATA_SHORT_XYZ: bpp = 6; break;
    }
    unsigned int cdoff = bpp;
    switch (cdType) {
    case SimpleSphericalParticles::COLDATA_NONE: break;
    case SimpleSphericalParticles::COLDATA_UINT8_RGB: bpp += 3; break;
    case SimpleSphericalParticles::COLDATA_UINT8_RGBA: bpp += 4; break;
    case SimpleSphericalParticles::COLDATA_FLOAT_RGB: bpp += 12; break;
    case SimpleSphericalParticles::COLDATA_FLOAT_RGBA: bpp += 16; break;
    case SimpleSphericalParticles::COLDATA_FLOAT_I: bpp += 4; break;
    }

    this->data.AssertSize(static_cast<SIZE_T>(partCnt * bpp));
    this->parts.SetVertexData(vdType, this->data.At(0), bpp);
    this->parts.SetColourData(cdType, this->data.At(cdoff), bpp);

    partCnt = 0;
    for (unsigned int li = 0; li < inDat.GetParticleListCount(); li++) {
        core::moldyn::MultiParticleDataCall::Particles& p = inDat.AccessParticles(li);
        if (p.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_NONE) continue;
        if (p.GetCount() == 0) continue;

        const uint8_t *pvd = static_cast<const uint8_t*>(p.GetVertexData());
        size_t pvds = p.GetVertexDataStride();
        switch (p.GetVertexDataType()) {
            case SimpleSphericalParticles::VERTDATA_NONE: throw std::exception();
            case SimpleSphericalParticles::VERTDATA_FLOAT_XYZ: pvds = std::max<size_t>(pvds, 12); break;
            case SimpleSphericalParticles::VERTDATA_FLOAT_XYZR: pvds = std::max<size_t>(pvds, 16); break;
            case SimpleSphericalParticles::VERTDATA_SHORT_XYZ: pvds = std::max<size_t>(pvds, 6); break;
        }
        const float pgr = p.GetGlobalRadius();

        const uint8_t * pcd = static_cast<const uint8_t*>(p.GetColourData());
        size_t pcds = p.GetColourDataStride();
        switch (p.GetColourDataType()) {
        case SimpleSphericalParticles::COLDATA_NONE: break;
        case SimpleSphericalParticles::COLDATA_UINT8_RGB: pcds = std::max<size_t>(pcds, 3); break;
        case SimpleSphericalParticles::COLDATA_UINT8_RGBA: pcds = std::max<size_t>(pcds, 4); break;
        case SimpleSphericalParticles::COLDATA_FLOAT_RGB: pcds = std::max<size_t>(pcds, 12); break;
        case SimpleSphericalParticles::COLDATA_FLOAT_RGBA: pcds = std::max<size_t>(pcds, 16); break;
        case SimpleSphericalParticles::COLDATA_FLOAT_I: pcds = std::max<size_t>(pcds, 4); break;
        }
        const uint8_t * const pgc = p.GetGlobalColour();
        const float pgcimin = p.GetMinColourIndexValue();
        const float pgcimax = p.GetMaxColourIndexValue();

        for (uint64_t pi = 0; pi < p.GetCount(); pi++, pcd += pcds, pvd += pvds, partCnt++) {
            // vertex data
            if (vdType == SimpleSphericalParticles::VERTDATA_SHORT_XYZ) {
                ASSERT(p.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_SHORT_XYZ);
                ::memcpy(this->data.At(static_cast<SIZE_T>(partCnt * bpp)), pvd, 6);
            } else {
                float *v = this->data.AsAt<float>(static_cast<SIZE_T>(partCnt * bpp));
                const float *pv = reinterpret_cast<const float*>(pvd);
                if (p.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_SHORT_XYZ) {
                    const uint16_t *pv = reinterpret_cast<const uint16_t*>(pvd);
                    v[0] = static_cast<float>(pv[0]);
                    v[1] = static_cast<float>(pv[1]);
                    v[2] = static_cast<float>(pv[2]);
                }
                ::memcpy(v, pv, 3 * sizeof(float));
                if (vdType == SimpleSphericalParticles::VERTDATA_FLOAT_XYZR) {
                    if (p.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZR) {
                        v[3] = pv[3];
                    } else {
                        v[3] = pgr;
                    }
                }
            }
            // color data
            if (cdType == SimpleSphericalParticles::COLDATA_NONE) continue;

            if ((cdType == SimpleSphericalParticles::COLDATA_UINT8_RGB)
                    || (cdType == SimpleSphericalParticles::COLDATA_UINT8_RGBA)) {
                ASSERT((p.GetColourDataType() == SimpleSphericalParticles::COLDATA_NONE)
                    || (p.GetColourDataType() == SimpleSphericalParticles::COLDATA_UINT8_RGB)
                    || (p.GetColourDataType() == SimpleSphericalParticles::COLDATA_UINT8_RGBA));
                uint8_t col[4];
                if (p.GetColourDataType() != SimpleSphericalParticles::COLDATA_NONE) {
                    ::memcpy(col, pcd, (p.GetColourDataType() == SimpleSphericalParticles::COLDATA_UINT8_RGB) ? 3 : 4);
                }
                ::memcpy(this->data.At(static_cast<SIZE_T>(partCnt * bpp + cdoff)),
                    (p.GetColourDataType() == SimpleSphericalParticles::COLDATA_NONE) ? pgc : col,
                    ((cdType == SimpleSphericalParticles::COLDATA_UINT8_RGBA) ? 4 : 3));
            }

            if ((cdType == SimpleSphericalParticles::COLDATA_FLOAT_RGB)
                    || (cdType == SimpleSphericalParticles::COLDATA_FLOAT_RGBA)) {
                // p.GetColourDataType() could be anything!
                float col[4];
                switch (p.GetColourDataType()) {
                case SimpleSphericalParticles::COLDATA_NONE:
                    col[0] = static_cast<float>(pgc[0]) / 255.0f;
                    col[1] = static_cast<float>(pgc[1]) / 255.0f;
                    col[2] = static_cast<float>(pgc[2]) / 255.0f;
                    col[3] = static_cast<float>(pgc[3]) / 255.0f;
                    break;
                case SimpleSphericalParticles::COLDATA_UINT8_RGBA: 
                    col[3] = static_cast<float>(reinterpret_cast<const uint8_t*>(pcd)[3]) / 255.0f;
                    // fall through
                case SimpleSphericalParticles::COLDATA_UINT8_RGB: 
                    col[2] = static_cast<float>(reinterpret_cast<const uint8_t*>(pcd)[2]) / 255.0f;
                    col[1] = static_cast<float>(reinterpret_cast<const uint8_t*>(pcd)[1]) / 255.0f;
                    col[0] = static_cast<float>(reinterpret_cast<const uint8_t*>(pcd)[0]) / 255.0f;
                    break;
                case SimpleSphericalParticles::COLDATA_FLOAT_RGBA:
                    col[3] = reinterpret_cast<const float*>(pcd)[3];
                    // fall through
                case SimpleSphericalParticles::COLDATA_FLOAT_RGB:
                    col[2] = reinterpret_cast<const float*>(pcd)[2];
                    col[1] = reinterpret_cast<const float*>(pcd)[1];
                    col[0] = reinterpret_cast<const float*>(pcd)[0];
                    break;
                case SimpleSphericalParticles::COLDATA_FLOAT_I: {
                    float colI = *reinterpret_cast<const float*>(pcd);
                    colI = (colI - pgcimin) / (pgcimax - pgcimin);
                    this->tfq.Query(col, colI);
                } break;
                }
                ::memcpy(this->data.At(static_cast<SIZE_T>(partCnt * bpp + cdoff)), col, sizeof(float) * ((cdType == SimpleSphericalParticles::COLDATA_FLOAT_RGBA) ? 4 : 3));
            }

            if (cdType == SimpleSphericalParticles::COLDATA_FLOAT_I) {
                ASSERT(p.GetColourDataType() == SimpleSphericalParticles::COLDATA_FLOAT_I);
                float col;
                col = *reinterpret_cast<const float*>(pcd);
                col = (col - pgcimin) / (pgcimax - pgcimin);
                col = col * (gcimax - gcimin) + gcimin;
                *this->data.AsAt<float>(static_cast<SIZE_T>(partCnt * bpp + cdoff)) = col;
            }
        }
    }

}
