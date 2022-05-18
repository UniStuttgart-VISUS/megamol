/*
 * ParticleBoxGeneratorDataSource.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "ParticleBoxGeneratorDataSource.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "stdafx.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <random>


using namespace megamol;
using namespace megamol::datatools;

namespace {
enum MyColorType : int { COLOR_NONE = 0, COLOR_RGBu8, COLOR_RGBAu8, COLOR_If, COLOR_RGBf, COLOR_RGBAf };
}


/*
 * ParticleBoxGeneratorDataSource::ParticleBoxGeneratorDataSource
 */
ParticleBoxGeneratorDataSource::ParticleBoxGeneratorDataSource(void)
        : core::Module()
        , dataSlot("data", "publishes the generated data")
        , randomSeedSlot("random::seed", "The random generator seed value")
        , randomReseedSlot("random::reseed", "Picks a new random seed value based on the current time")
        , particleCountSlot("count", "Number of particles to be generated")
        , radiusPerParticleSlot("store::explicitRadius", "Flag to explicitly store radii at each particle")
        , colorDataSlot("store::color", "Type of color information to be generated")
        , interleavePosAndColorSlot("store::interleaved", "Flag to interleave position and color information")
        , radiusScaleSlot("radiusScale", "Scale factor for particle radii")
        , positionNoiseSlot("positionNoise", "Amount of noise for the position values")
        , dataHash(0)
        , cnt(0)
        , data()
        , rad(0.0f)
        , vdt(Particles::VERTDATA_NONE)
        , vdp(nullptr)
        , vds(0)
        , cdt(Particles::COLDATA_NONE)
        , cdp(nullptr)
        , cds(0) {

    dataSlot.SetCallback("MultiParticleDataCall", "GetData", &ParticleBoxGeneratorDataSource::getDataCallback);
    dataSlot.SetCallback("MultiParticleDataCall", "GetExtent", &ParticleBoxGeneratorDataSource::getExtentCallback);
    MakeSlotAvailable(&dataSlot);

    particleCountSlot.SetParameter(new core::param::IntParam(1000, 0));
    MakeSlotAvailable(&particleCountSlot);

    radiusPerParticleSlot.SetParameter(new core::param::BoolParam(false));
    MakeSlotAvailable(&radiusPerParticleSlot);

    core::param::EnumParam* colType = new core::param::EnumParam(0);
    colType->SetTypePair(COLOR_NONE, "none");
    colType->SetTypePair(COLOR_RGBu8, "RGB (bytes)");
    colType->SetTypePair(COLOR_RGBAu8, "RGBA (bytes)");
    colType->SetTypePair(COLOR_If, "Lum (float)");
    colType->SetTypePair(COLOR_RGBf, "RGB (floats)");
    colType->SetTypePair(COLOR_RGBAf, "RGBA (floats)");
    colorDataSlot.SetParameter(colType);
    MakeSlotAvailable(&colorDataSlot);

    interleavePosAndColorSlot.SetParameter(new core::param::BoolParam(true));
    MakeSlotAvailable(&interleavePosAndColorSlot);

    radiusScaleSlot.SetParameter(new core::param::FloatParam(0.5f, 0.0f));
    MakeSlotAvailable(&radiusScaleSlot);

    positionNoiseSlot.SetParameter(new core::param::FloatParam(1.25f, 0.0f));
    MakeSlotAvailable(&positionNoiseSlot);

    randomSeedSlot.SetParameter(new core::param::IntParam(2007, 0));
    randomSeedSlot.ForceSetDirty();
    MakeSlotAvailable(&randomSeedSlot);

    randomReseedSlot.SetParameter(new core::param::ButtonParam());
    randomReseedSlot.SetUpdateCallback(&ParticleBoxGeneratorDataSource::reseed);
    MakeSlotAvailable(&randomReseedSlot);
}


/*
 * ParticleBoxGeneratorDataSource::~ParticleBoxGeneratorDataSource
 */
ParticleBoxGeneratorDataSource::~ParticleBoxGeneratorDataSource(void) {
    this->Release();
}


/*
 * ParticleBoxGeneratorDataSource::create
 */
bool ParticleBoxGeneratorDataSource::create(void) {
    // intentionally empty
    return true;
}


/*
 * ParticleBoxGeneratorDataSource::release
 */
void ParticleBoxGeneratorDataSource::release(void) {
    this->clear();
}


/*
 * ParticleBoxGeneratorDataSource::reseed
 */
bool ParticleBoxGeneratorDataSource::reseed(core::param::ParamSlot& p) {
    std::mt19937 rnd_engine(
        static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<int> rnd_uni_int(0, 1024 * 1024);

    randomSeedSlot.Param<core::param::IntParam>()->SetValue(rnd_uni_int(rnd_engine));

    return true; // reset dirty flag
}


/*
 * ParticleBoxGeneratorDataSource::getDataCallback
 */
bool ParticleBoxGeneratorDataSource::getDataCallback(core::Call& caller) {
    geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&caller);
    if (mpdc == nullptr)
        return false;
    if (particleCountSlot.IsDirty() || radiusPerParticleSlot.IsDirty() || colorDataSlot.IsDirty() ||
        interleavePosAndColorSlot.IsDirty() || radiusScaleSlot.IsDirty() || positionNoiseSlot.IsDirty() ||
        randomSeedSlot.IsDirty()) {
        this->assertData();
    }

    mpdc->SetParticleListCount(1);
    mpdc->SetDataHash(dataHash);
    mpdc->SetExtent(1, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    geocalls::MultiParticleDataCall::Particles& parties = mpdc->AccessParticles(0);

    parties.SetCount(cnt);
    parties.SetGlobalRadius(rad);
    parties.SetGlobalColour(127, 127, 127);
    parties.SetColourMapIndexValues(0.0f, 1.0f);
    parties.SetVertexData(vdt, vdp, vds);
    parties.SetColourData(cdt, cdp, cds);

    mpdc->SetUnlocker(nullptr);

    return true;
}


/*
 * ParticleBoxGeneratorDataSource::getExtentCallback
 */
bool ParticleBoxGeneratorDataSource::getExtentCallback(core::Call& caller) {
    geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&caller);
    if (mpdc == nullptr)
        return false;
    if (particleCountSlot.IsDirty() || radiusPerParticleSlot.IsDirty() || colorDataSlot.IsDirty() ||
        interleavePosAndColorSlot.IsDirty() || radiusScaleSlot.IsDirty() || positionNoiseSlot.IsDirty() ||
        randomSeedSlot.IsDirty()) {
        this->assertData();
    }

    mpdc->SetParticleListCount(1);
    mpdc->SetDataHash(dataHash);
    mpdc->SetExtent(1, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);

    mpdc->SetUnlocker(nullptr);

    return true;
}


/*
 * ParticleBoxGeneratorDataSource::clear
 */
void ParticleBoxGeneratorDataSource::clear(void) {
    cnt = 0;
    data.EnforceSize(0);
    rad = 0.0f;
    vdt = Particles::VERTDATA_NONE;
    vdp = nullptr;
    vds = 0;
    cdt = Particles::COLDATA_NONE;
    cdp = nullptr;
    cds = 0;
}


/*
 * ParticleBoxGeneratorDataSource::assertData
 */
void ParticleBoxGeneratorDataSource::assertData(void) {
    dataHash++;

    randomSeedSlot.ResetDirty();
    particleCountSlot.ResetDirty();
    radiusPerParticleSlot.ResetDirty();
    colorDataSlot.ResetDirty();
    interleavePosAndColorSlot.ResetDirty();
    radiusScaleSlot.ResetDirty();
    positionNoiseSlot.ResetDirty();

    uint64_t pcnt = particleCountSlot.Param<core::param::IntParam>()->Value();
    bool radAtPart = radiusPerParticleSlot.Param<core::param::BoolParam>()->Value();
    MyColorType colType = static_cast<MyColorType>(colorDataSlot.Param<core::param::EnumParam>()->Value());
    float radScale = radiusScaleSlot.Param<core::param::FloatParam>()->Value();
    float posNoise = positionNoiseSlot.Param<core::param::FloatParam>()->Value();

    if (pcnt == 0) {
        clear();
        return;
    }

    // initialize debugger and distribution
    std::mt19937 rnd_engine(static_cast<uint32_t>(randomSeedSlot.Param<core::param::IntParam>()->Value()));
    std::uniform_real_distribution<float> rnd_uni; // [0, 1]
    std::uniform_int_distribution<int> rnd_uni_int(0, 255);

    // setup storage and pointer
    int bpv = 12;
    int bpc = 0;
    if (radAtPart) {
        vdt = Particles::VERTDATA_FLOAT_XYZR;
        bpv += 4;
    } else {
        vdt = Particles::VERTDATA_FLOAT_XYZ;
    }
    switch (colType) {
    case COLOR_NONE:
        cdt = Particles::COLDATA_NONE;
        break;
    case COLOR_RGBu8:
        cdt = Particles::COLDATA_UINT8_RGB;
        bpc = 3;
        break;
    case COLOR_RGBAu8:
        cdt = Particles::COLDATA_UINT8_RGBA;
        bpc = 4;
        break;
    case COLOR_If:
        cdt = Particles::COLDATA_FLOAT_I;
        bpc = 4;
        break;
    case COLOR_RGBf:
        cdt = Particles::COLDATA_FLOAT_RGB;
        bpc = 12;
        break;
    case COLOR_RGBAf:
        cdt = Particles::COLDATA_FLOAT_RGBA;
        bpc = 16;
        break;
    default:
        cdt = Particles::COLDATA_NONE;
        colType = COLOR_NONE;
        break;
    }

    // prepare memory write positions and position increments depending on the layout
    data.EnforceSize(static_cast<size_t>(pcnt * (bpv + bpc)));
    vdp = data.AsAt<void>(0);
    if (interleavePosAndColorSlot.Param<core::param::BoolParam>()->Value()) {
        cdp = data.AsAt<void>(bpv);
        vds = cds = bpv + bpc;
    } else {
        cdp = data.AsAt<void>(static_cast<size_t>(bpv * pcnt));
        vds = bpv;
        cds = bpc;
    }
    unsigned char* vertDatPtr = static_cast<unsigned char*>(vdp);
    unsigned char* colDatPtr = static_cast<unsigned char*>(cdp);

    // Compute box layout
    uint32_t xcnt, ycnt, zcnt;
    float yzcnt;
    xcnt = static_cast<uint32_t>(std::ceil(static_cast<float>(std::pow(static_cast<double>(pcnt), 1.0 / 3.0))));
    yzcnt = std::ceil(static_cast<float>(pcnt) / static_cast<float>(xcnt));
    ycnt = static_cast<uint32_t>(std::ceil(std::sqrt(yzcnt)));
    zcnt = static_cast<uint32_t>(std::ceil(yzcnt / static_cast<float>(ycnt)));
    //std::cout << "Boxing: " << xcnt << " x " << ycnt << " x " << zcnt << std::endl;

    float x_a = 2.0f / static_cast<float>(xcnt);
    float x_b = x_a / 2.0f;
    float y_a = 2.0f / static_cast<float>(ycnt);
    float y_b = y_a / 2.0f;
    float z_a = 2.0f / static_cast<float>(zcnt);
    float z_b = z_a / 2.0f;

    rad = std::min<float>(std::min<float>(x_b, y_b), z_b) * radScale;
    float pn = rad * posNoise;

    uint32_t i_all = 0;
    for (uint32_t iz = 0; (iz < zcnt) && (i_all < pcnt); ++iz) {
        float pz = -1.0f + z_b + z_a * iz;
        for (uint32_t iy = 0; (iy < ycnt) && (i_all < pcnt); ++iy) {
            float py = -1.0f + y_b + y_a * iy;
            for (uint32_t ix = 0; (ix < xcnt) && (i_all < pcnt); ++ix, ++i_all) {
                float px = -1.0f + x_b + x_a * ix;

                // position
                float* vertDat = reinterpret_cast<float*>(vertDatPtr);
                vertDat[0] = px + (rnd_uni(rnd_engine) * 2.0f - 1.0f) * pn;
                vertDat[1] = py + (rnd_uni(rnd_engine) * 2.0f - 1.0f) * pn;
                vertDat[2] = pz + (rnd_uni(rnd_engine) * 2.0f - 1.0f) * pn;
                if (radAtPart)
                    vertDat[3] = rad;

                // color
                switch (colType) {
                case COLOR_RGBu8: {
                    unsigned char* colDat = reinterpret_cast<unsigned char*>(colDatPtr);
                    colDat[0] = rnd_uni_int(rnd_engine);
                    colDat[1] = rnd_uni_int(rnd_engine);
                    colDat[2] = rnd_uni_int(rnd_engine);
                } break;
                case COLOR_RGBAu8: {
                    unsigned char* colDat = reinterpret_cast<unsigned char*>(colDatPtr);
                    colDat[0] = rnd_uni_int(rnd_engine);
                    colDat[1] = rnd_uni_int(rnd_engine);
                    colDat[2] = rnd_uni_int(rnd_engine);
                    colDat[3] = rnd_uni_int(rnd_engine);
                } break;
                case COLOR_If: {
                    float* colDat = reinterpret_cast<float*>(colDatPtr);
                    colDat[0] = rnd_uni(rnd_engine);
                } break;
                case COLOR_RGBf: {
                    float* colDat = reinterpret_cast<float*>(colDatPtr);
                    colDat[0] = rnd_uni(rnd_engine);
                    colDat[1] = rnd_uni(rnd_engine);
                    colDat[2] = rnd_uni(rnd_engine);
                } break;
                case COLOR_RGBAf: {
                    float* colDat = reinterpret_cast<float*>(colDatPtr);
                    colDat[0] = rnd_uni(rnd_engine);
                    colDat[1] = rnd_uni(rnd_engine);
                    colDat[2] = rnd_uni(rnd_engine);
                    colDat[3] = rnd_uni(rnd_engine);
                } break;
                }

                vertDatPtr += vds;
                colDatPtr += cds;
            }
        }
    }

    cnt = pcnt;
}
