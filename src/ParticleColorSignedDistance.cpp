/*
 * ParticleColorSignedDistance.h
 *
 * Copyright (C) 2015 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#ifdef WITH_ANN
#include "ParticleColorSignedDistance.h"
#include "mmcore/param/BoolParam.h"
#include "ANN/ANN.h"
#include <cstdint>
#include <algorithm>
#include <cfloat>
#include <cassert>

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::ParticleColorSignedDistance::ParticleColorSignedDistance
 */
datatools::ParticleColorSignedDistance::ParticleColorSignedDistance(void)
        : AbstractParticleManipulator("outData", "indata"),
        enableSlot("enable", "Enables the color manipulation"),
        cyclXSlot("cyclX", "Considders cyclic boundary conditions in X direction"),
        cyclYSlot("cyclY", "Considders cyclic boundary conditions in Y direction"),
        cyclZSlot("cyclZ", "Considders cyclic boundary conditions in Z direction"),
        datahash(0), time(0), newColors(), minCol(0.0f), maxCol(1.0f) {

    this->enableSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->enableSlot);

    this->cyclXSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclXSlot);

    this->cyclYSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclYSlot);

    this->cyclZSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclZSlot);
}


/*
 * datatools::ParticleColorSignedDistance::~ParticleColorSignedDistance
 */
datatools::ParticleColorSignedDistance::~ParticleColorSignedDistance(void) {
    this->Release();
}


/*
 * datatools::ParticleColorSignedDistance::manipulateData
 */
bool datatools::ParticleColorSignedDistance::manipulateData(
        megamol::core::moldyn::MultiParticleDataCall& outData,
        megamol::core::moldyn::MultiParticleDataCall& inData) {
    using megamol::core::moldyn::MultiParticleDataCall;

    outData = inData; // also transfers the unlocker to 'outData'
    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    if (!this->enableSlot.Param<core::param::BoolParam>()->Value()) return true;

    if (this->cyclXSlot.IsDirty()) {
        this->cyclXSlot.ResetDirty();
        this->datahash = 0;
    }
    if (this->cyclYSlot.IsDirty()) {
        this->cyclYSlot.ResetDirty();
        this->datahash = 0;
    }
    if (this->cyclZSlot.IsDirty()) {
        this->cyclZSlot.ResetDirty();
        this->datahash = 0;
    }
    if ((this->datahash == 0) || (this->datahash != outData.DataHash()) || (this->time != outData.FrameID())) {
        this->datahash = outData.DataHash();
        this->time = outData.FrameID();
        this->compute_colors(outData);
    }
    
    if (this->newColors.size() > 0) {
        this->set_colors(outData);
    }

    return true;
}


void datatools::ParticleColorSignedDistance::compute_colors(megamol::core::moldyn::MultiParticleDataCall& dat) {
    using megamol::core::moldyn::SimpleSphericalParticles;
    size_t allpartcnt = 0;
    size_t negpartcnt = 0;
    size_t nulpartcnt = 0;
    size_t pospartcnt = 0;
    const float border_epsilon = 0.001f;

    // count particles
    unsigned int plc = dat.GetParticleListCount();
    for (unsigned int pli = 0; pli < plc; pli++) {
        auto& pl = dat.AccessParticles(pli);
        if (pl.GetColourDataType() != SimpleSphericalParticles::COLDATA_FLOAT_I) continue;
        if ((pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZ)
            && (pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)) {
            continue;
        }
        allpartcnt += static_cast<size_t>(pl.GetCount());
    }

    this->newColors.resize(allpartcnt);
    ANNpoint dataPtsData = new ANNcoord[3 * allpartcnt];
    std::vector<size_t> posparts;
    std::vector<size_t> negparts;
    posparts.reserve(allpartcnt);
    negparts.reserve(allpartcnt);

    allpartcnt = 0;
    for (unsigned int pli = 0; pli < plc; pli++) {
        auto& pl = dat.AccessParticles(pli);
        if (pl.GetColourDataType() != SimpleSphericalParticles::COLDATA_FLOAT_I) continue;
        if ((pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZ)
            && (pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)) {
            continue;
        }

        unsigned int vert_stride = 0;
        if (pl.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZ) vert_stride = 12;
        else if (pl.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZR) vert_stride = 16;
        else continue;
        vert_stride = std::max<unsigned int>(vert_stride, pl.GetVertexDataStride());
        const unsigned char *vert = static_cast<const unsigned char*>(pl.GetVertexData());

        int part_cnt = static_cast<int>(pl.GetCount());
        const unsigned char *col = static_cast<const unsigned char*>(pl.GetColourData());
        unsigned int stride = std::max<unsigned int>(pl.GetColourDataStride(), sizeof(float));

        for (int part_i = 0; part_i < part_cnt; ++part_i) {
            float c = *reinterpret_cast<const float *>(col + (part_i * stride));
            const float *v = reinterpret_cast<const float *>(vert + (part_i * vert_stride));
            dataPtsData[(allpartcnt + part_i) * 3 + 0] = static_cast<ANNcoord>(v[0]);
            dataPtsData[(allpartcnt + part_i) * 3 + 1] = static_cast<ANNcoord>(v[1]);
            dataPtsData[(allpartcnt + part_i) * 3 + 2] = static_cast<ANNcoord>(v[2]);

            if (c < -border_epsilon) {
                negpartcnt++;
                negparts.push_back(allpartcnt + part_i);
            } else if (c < border_epsilon) {
                nulpartcnt++;
                negparts.push_back(allpartcnt + part_i);
                posparts.push_back(allpartcnt + part_i);
            } else {
                pospartcnt++;
                posparts.push_back(allpartcnt + part_i);
            }
        }
        allpartcnt += static_cast<size_t>(pl.GetCount());
    }

    // allocate ANN data structures for border
    assert(pospartcnt + nulpartcnt == posparts.size());
    ANNpointArray posnulPts = new ANNpoint[posparts.size()];
    for (size_t i = 0; i < pospartcnt + nulpartcnt; ++i) {
        posnulPts[i] = dataPtsData + (posparts[i] * 3);
    }
    posparts.clear();
    assert(negpartcnt + nulpartcnt == negparts.size());
    ANNpointArray negnulPts = new ANNpoint[negparts.size()];
    for (size_t i = 0; i < negpartcnt + nulpartcnt; ++i) {
        negnulPts[i] = dataPtsData + (negparts[i] * 3);
    }
    negparts.clear();
    ANNkd_tree* posTree = new ANNkd_tree(posnulPts, static_cast<int>(pospartcnt + nulpartcnt), 3);
    ANNkd_tree* negTree = new ANNkd_tree(negnulPts, static_cast<int>(negpartcnt + nulpartcnt), 3);

    // final computation
    allpartcnt = 0;
    for (unsigned int pli = 0; pli < plc; pli++) {
        auto& pl = dat.AccessParticles(pli);
        if (pl.GetColourDataType() != SimpleSphericalParticles::COLDATA_FLOAT_I) continue;
        if ((pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZ)
            && (pl.GetVertexDataType() != SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)) {
            continue;
        }

        unsigned int vert_stride = 0;
        if (pl.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZ) vert_stride = 12;
        else if (pl.GetVertexDataType() == SimpleSphericalParticles::VERTDATA_FLOAT_XYZR) vert_stride = 16;
        else continue;
        vert_stride = std::max<unsigned int>(vert_stride, pl.GetVertexDataStride());
        const unsigned char *vert = static_cast<const unsigned char*>(pl.GetVertexData());

        int part_cnt = static_cast<int>(pl.GetCount());
        const unsigned char *col = static_cast<const unsigned char*>(pl.GetColourData());
        unsigned int col_stride = std::max<unsigned int>(pl.GetColourDataStride(), sizeof(float));

        for (int part_i = 0; part_i < part_cnt; ++part_i) {
            float c = *reinterpret_cast<const float *>(col + (part_i * col_stride));
            const float *v = reinterpret_cast<const float *>(vert + (part_i * vert_stride));
            ANNcoord q[3] = {
                static_cast<ANNcoord>(v[0]),
                static_cast<ANNcoord>(v[1]),
                static_cast<ANNcoord>(v[2])};
            ANNidx ni;
            ANNdist nd;

            if (c < -border_epsilon) {
                // neg
                posTree->annkSearch(q, 1, &ni, &nd);
                nd = /*-*/sqrt(nd);
                c = static_cast<float>(nd);

            } else if (c < border_epsilon) {
                // null
                c = 0.0f;

            } else {
                // pos
                negTree->annkSearch(q, 1, &ni, &nd);
                nd = sqrt(nd);
                c = static_cast<float>(nd);
            }

            if (c < this->minCol) this->minCol = c;
            if (c > this->maxCol) this->maxCol = c;

            this->newColors[allpartcnt + part_i] = c;
        }

        allpartcnt += static_cast<size_t>(part_cnt);
    }
    
    delete posTree;
    delete negTree;
    delete[] posnulPts;
    delete[] negnulPts;
    delete[] dataPtsData;
}


void datatools::ParticleColorSignedDistance::set_colors(megamol::core::moldyn::MultiParticleDataCall& dat) {
    size_t allpartcnt = 0;

    unsigned int plc = dat.GetParticleListCount();
    for (unsigned int pli = 0; pli < plc; pli++) {
        auto& pl = dat.AccessParticles(pli);
        if (pl.GetColourDataType() != megamol::core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I) continue;

        pl.SetColourData(megamol::core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, this->newColors.data() + allpartcnt);
        pl.SetColourMapIndexValues(this->minCol, this->maxCol);

        allpartcnt += static_cast<size_t>(pl.GetCount());
    }
}
#endif