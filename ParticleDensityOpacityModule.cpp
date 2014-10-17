#include "stdafx.h"
#include "ParticleDensityOpacityModule.h"
#include "view/CallGetTransferFunction.h"
#include "param/ButtonParam.h"
#include "param/IntParam.h"
#include "param/FloatParam.h"
#include "param/BoolParam.h"
#include "vislib/ColourRGBAu8.h"
#include "vislib/IncludeAllGL.h"
#include <cstdint>
#include "vislib/ShallowPoint.h"
#include "ANN/ANN.h"


megamol::stdplugin::datatools::ParticleDensityOpacityModule::ParticleDensityOpacityModule(void) : Module(),
        putDataSlot("putdata", "Connects from the data consumer"),
        getDataSlot("getdata", "Connects to the data source"),
        getTFSlot("gettransferfunction", "Connects to the transfer function module"),
        rebuildButtonSlot("rebuild", "Forces rebuild of colour data"),
        densityRadiusSlot("density::radius", "The radius of the local volume for the density computation"),
        densityMinCountSlot("density::min", "The minimum density particle count"),
        densityMaxCountSlot("density::max", "The maximum density particle count"),
        densityComputeCountRangeSlot("density::computeRange", "Compute 'min' and 'max' values automatically"),
        opacityMinValSlot("opacity::min", "Minimum opacity value"),
        opacityMaxValSlot("opacity::max", "Maximum opacity value"),
        cyclBoundXSlot("periodicBoundary::x", "Dis-/Enables periodic boundary conditions in x direction"),
        cyclBoundYSlot("periodicBoundary::y", "Dis-/Enables periodic boundary conditions in y direction"),
        cyclBoundZSlot("periodicBoundary::z", "Dis-/Enables periodic boundary conditions in z direction"),
        mapDensityToAlphaSlot("opacity::mapAlpha", "Maps the opacity to the color alpha"),
        mapDensityToColorSlot("opacity::mapColor", "Maps the opacity to the color"),
        lastFrame(0), lastHash(0), colData() {

    this->putDataSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(), "GetData", &ParticleDensityOpacityModule::getDataCallback);
    this->putDataSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(), "GetExtent", &ParticleDensityOpacityModule::getExtentCallback);
    this->MakeSlotAvailable(&this->putDataSlot);

    this->getDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->rebuildButtonSlot.SetParameter(new core::param::ButtonParam());
    this->MakeSlotAvailable(&this->rebuildButtonSlot);

    this->densityRadiusSlot.SetParameter(new core::param::FloatParam(10.0f, 0.0f));
    this->MakeSlotAvailable(&this->densityRadiusSlot);

    this->densityMinCountSlot.SetParameter(new core::param::IntParam(0, 0));
    this->MakeSlotAvailable(&this->densityMinCountSlot);

    this->densityMaxCountSlot.SetParameter(new core::param::IntParam(100, 0));
    this->MakeSlotAvailable(&this->densityMaxCountSlot);

    this->densityComputeCountRangeSlot.SetParameter(new core::param::ButtonParam());
    this->MakeSlotAvailable(&this->densityComputeCountRangeSlot);

    this->opacityMinValSlot.SetParameter(new core::param::FloatParam(0.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->opacityMinValSlot);

    this->opacityMaxValSlot.SetParameter(new core::param::FloatParam(1.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->opacityMaxValSlot);

    this->cyclBoundXSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclBoundXSlot);

    this->cyclBoundYSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclBoundYSlot);

    this->cyclBoundZSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclBoundZSlot);

    this->mapDensityToAlphaSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->mapDensityToAlphaSlot);

    this->mapDensityToColorSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->mapDensityToColorSlot);
}


megamol::stdplugin::datatools::ParticleDensityOpacityModule::~ParticleDensityOpacityModule(void) {
    this->Release();
}


bool megamol::stdplugin::datatools::ParticleDensityOpacityModule::create(void) {
    // intentionally empty
    return true;
}


void megamol::stdplugin::datatools::ParticleDensityOpacityModule::release(void) {
    // intentionally empty
}


bool megamol::stdplugin::datatools::ParticleDensityOpacityModule::getDataCallback(core::Call& caller) {
    using ::megamol::core::moldyn::MultiParticleDataCall;
    MultiParticleDataCall *inCall = dynamic_cast<MultiParticleDataCall*>(&caller);
    if (inCall == nullptr) return false;

    MultiParticleDataCall *outCall = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if (outCall == nullptr) return false;

    *outCall = *inCall;

    if ((*outCall)(0)) {
        bool update_data(false);

        if ((this->lastFrame != outCall->FrameID()) || (this->lastHash != outCall->DataHash()) || (outCall->DataHash() == 0)) {
            this->lastFrame = outCall->FrameID();
            this->lastHash = outCall->DataHash();
            update_data = true;
        }

        if (this->rebuildButtonSlot.IsDirty()) {
            this->rebuildButtonSlot.ResetDirty();
            update_data = true;
        }
        if (this->densityComputeCountRangeSlot.IsDirty()) {
            update_data = true;
        }
        if (this->densityRadiusSlot.IsDirty() || this->densityMinCountSlot.IsDirty() || this->densityMaxCountSlot.IsDirty()) {
            this->densityRadiusSlot.ResetDirty();
            this->densityMinCountSlot.ResetDirty();
            this->densityMaxCountSlot.ResetDirty();
            update_data = true;
        }
        if (this->opacityMinValSlot.IsDirty() || this->opacityMaxValSlot.IsDirty()) {
            this->opacityMinValSlot.ResetDirty();
            this->opacityMaxValSlot.ResetDirty();
            update_data = true;
        }
        if (this->cyclBoundXSlot.IsDirty() || this->cyclBoundYSlot.IsDirty() || this->cyclBoundZSlot.IsDirty()) {
            this->cyclBoundXSlot.ResetDirty();
            this->cyclBoundYSlot.ResetDirty();
            this->cyclBoundZSlot.ResetDirty();
            update_data = true;
        }
        if (this->mapDensityToAlphaSlot.IsDirty() || this->mapDensityToColorSlot.IsDirty()) {
            this->mapDensityToAlphaSlot.ResetDirty();
            this->mapDensityToColorSlot.ResetDirty();
            update_data = true;
        }

        if (update_data) {
            this->makeData(outCall);
        }

        this->densityComputeCountRangeSlot.ResetDirty();

        *inCall = *outCall;

        size_t cnt = 0;
        unsigned int plc = inCall->GetParticleListCount();
        for (unsigned int pli = 0; pli < plc; pli++) {
            core::moldyn::SimpleSphericalParticles &p = inCall->AccessParticles(pli);
            if ((p.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_NONE)
                || (p.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_SHORT_XYZ)) continue;
            p.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA,
                this->colData.At(cnt * sizeof(float) * 4));
            cnt += static_cast<size_t>(p.GetCount());
        }

        inCall->SetUnlocker(new Unlocker(outCall->GetUnlocker()), false);
        outCall->SetUnlocker(nullptr, false);

        return true;
    }

    return false;
}


bool megamol::stdplugin::datatools::ParticleDensityOpacityModule::getExtentCallback(core::Call& caller) {
    core::moldyn::MultiParticleDataCall *inCall = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);
    if (inCall == nullptr) return false;

    core::moldyn::MultiParticleDataCall *outCall = this->getDataSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (outCall == nullptr) return false;

    *outCall = *inCall;

    if ((*outCall)(1)) {
        outCall->SetUnlocker(nullptr, false);
        *inCall = *outCall;
        return true;
    }

    return false;
}


void megamol::stdplugin::datatools::ParticleDensityOpacityModule::makeData(core::moldyn::MultiParticleDataCall *dat) {
    bool autoScale = this->densityComputeCountRangeSlot.IsDirty();

    size_t all_cnt = 0;
    unsigned int plc = dat->GetParticleListCount();
    for (unsigned int pli = 0; pli < plc; pli++) {
        core::moldyn::SimpleSphericalParticles::VertexDataType vdt = dat->AccessParticles(pli).GetVertexDataType();
        if ((vdt == core::moldyn::SimpleSphericalParticles::VERTDATA_NONE)
            || (vdt == core::moldyn::SimpleSphericalParticles::VERTDATA_SHORT_XYZ)) continue;
        all_cnt += static_cast<size_t>(dat->AccessParticles(pli).GetCount());
    }

    this->colData.EnforceSize(all_cnt * sizeof(float) * 4); // always store COLDATA_FLOAT_RGBA

    // first copy all color values
    vislib::RawStorage texDat;

    core::view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<core::view::CallGetTransferFunction>();
    if ((cgtf != NULL) && ((*cgtf)(0))) {
        ::glGetError();
        ::glEnable(GL_TEXTURE_1D);
        ::glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());

        int texSize = 0;
        ::glGetTexLevelParameteriv(GL_TEXTURE_1D, 0, GL_TEXTURE_WIDTH, &texSize);

        if (::glGetError() == GL_NO_ERROR) {
            texDat.EnforceSize(texSize * 12);
            ::glGetTexImage(GL_TEXTURE_1D, 0, GL_RGB, GL_FLOAT, texDat.As<void>());
            if (::glGetError() != GL_NO_ERROR) {
                texDat.EnforceSize(0);
            }
        }

        ::glBindTexture(GL_TEXTURE_1D, 0);
        ::glDisable(GL_TEXTURE_1D);
    }
    unsigned int texDatSize = 2;
    if (texDat.GetSize() < 24) {
        texDat.EnforceSize(24);
        *texDat.AsAt<float>(0) = 0.0f;
        *texDat.AsAt<float>(4) = 0.0f;
        *texDat.AsAt<float>(8) = 0.0f;
        *texDat.AsAt<float>(12) = 1.0f;
        *texDat.AsAt<float>(16) = 1.0f;
        *texDat.AsAt<float>(20) = 1.0f;
    } else {
        texDatSize = static_cast<unsigned int>(texDat.GetSize() / 12);
    }

    ANNpointArray dataPts = new ANNpoint[all_cnt];
    ANNpoint dataPtsData = new ANNcoord[3 * all_cnt];

    size_t ci = 0;
    float *f = this->colData.As<float>();
    for (unsigned int pli = 0; pli < plc; pli++) {
        core::moldyn::MultiParticleDataCall::Particles &pl = dat->AccessParticles(pli);
        if ((pl.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_NONE)
            || (pl.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_SHORT_XYZ)) continue;
        size_t col_stride = 0;
        bool bytes = true;
        const uint8_t *pld = static_cast<const uint8_t*>(pl.GetColourData());
        size_t vert_stride = pl.GetVertexDataStride();
        const uint8_t *vert = static_cast<const uint8_t*>(pl.GetVertexData());
        switch (pl.GetVertexDataType()) {
        case core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ:
            if (vert_stride < 12) vert_stride = 12;
            break;
        case core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR:
            if (vert_stride < 16) vert_stride = 16;
            break;
        default: throw std::exception();
        }

        for (uint64_t pi = 0; pi < pl.GetCount(); pi++, vert += vert_stride) {
            size_t idx = static_cast<size_t>(pi + ci);
            dataPts[idx] = dataPtsData + idx * 3;
            dataPtsData[idx * 3 + 0] = static_cast<ANNcoord>(reinterpret_cast<const float*>(vert)[0]);
            dataPtsData[idx * 3 + 1] = static_cast<ANNcoord>(reinterpret_cast<const float*>(vert)[1]);
            dataPtsData[idx * 3 + 2] = static_cast<ANNcoord>(reinterpret_cast<const float*>(vert)[2]);
        }

        switch (pl.GetColourDataType()) {
        case core::moldyn::SimpleSphericalParticles::COLDATA_NONE: { //< use global colour
            float r = static_cast<float>(pl.GetGlobalColour()[0]) / 255.0f;
            float g = static_cast<float>(pl.GetGlobalColour()[1]) / 255.0f;
            float b = static_cast<float>(pl.GetGlobalColour()[2]) / 255.0f;
            for (uint64_t pi = 0; pi < pl.GetCount(); pi++, ci++) {
                f[ci * 4 + 0] = r;
                f[ci * 4 + 1] = g;
                f[ci * 4 + 2] = b;
            }
        } continue;
        case core::moldyn::SimpleSphericalParticles::COLDATA_UINT8_RGB: bytes = true; col_stride = 3; break;
        case core::moldyn::SimpleSphericalParticles::COLDATA_UINT8_RGBA: bytes = true; col_stride = 4; break;
        case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGB: bytes = false; col_stride = 12; break;
        case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA: bytes = false; col_stride = 16; break;
        case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I: { //< single float value to be mapped by a transfer function
            col_stride = (4 < pl.GetColourDataStride()) ? pl.GetColourDataStride() : 4;
            float cvmin = pl.GetMinColourIndexValue();
            float cvrng = pl.GetMaxColourIndexValue() - cvmin;
            if (cvrng <= 0.0f) cvrng = 0.0f; else cvrng = 1.0f / cvrng;
            for (uint64_t pi = 0; pi < pl.GetCount(); pi++, ci++, pld += col_stride) {
                const float *plf = reinterpret_cast<const float*>(pld);
                float c = (*plf - cvmin) * cvrng; // color value in [0..1]
                c *= static_cast<float>(texDatSize);
                vislib::math::Point<float, 3> cc;
                if (c >= static_cast<float>(texDatSize)) cc = vislib::math::ShallowPoint<float, 3>(texDat.AsAt<float>((texDatSize - 1) * 3 * sizeof(float)));
                else {
                    int cidx = static_cast<int>(c);
                    vislib::math::ShallowPoint<float, 3> c0(texDat.AsAt<float>(cidx * 3 * sizeof(float)));
                    vislib::math::ShallowPoint<float, 3> c1(texDat.AsAt<float>((cidx + 1) * 3 * sizeof(float)));
                    cc = c0.Interpolate(c1, c - static_cast<float>(cidx));
                }
                f[ci * 4 + 0] = cc[0];
                f[ci * 4 + 1] = cc[1];
                f[ci * 4 + 2] = cc[2];
            }
        } continue;
        }
        if (col_stride < pl.GetColourDataStride()) col_stride = pl.GetColourDataStride();
        if (bytes) {
            for (uint64_t pi = 0; pi < pl.GetCount(); pi++, ci++, pld += col_stride) {
                const uint8_t *plb = reinterpret_cast<const uint8_t*>(pld);
                f[ci * 4 + 0] = static_cast<float>(plb[0]) / 255.0f;
                f[ci * 4 + 1] = static_cast<float>(plb[1]) / 255.0f;
                f[ci * 4 + 2] = static_cast<float>(plb[2]) / 255.0f;
            }
        } else {
            for (uint64_t pi = 0; pi < pl.GetCount(); pi++, ci++, pld += col_stride) {
                const float *plf = reinterpret_cast<const float*>(pld);
                f[ci * 4 + 0] = plf[0];
                f[ci * 4 + 1] = plf[1];
                f[ci * 4 + 2] = plf[2];
            }
        }
    }

    ANNkd_tree* kdTree = new ANNkd_tree(dataPts, static_cast<int>(all_cnt), 3);
    ANNdist rad_sq = static_cast<ANNdist>(this->densityRadiusSlot.Param<core::param::FloatParam>()->Value());
    rad_sq *= rad_sq;

    int minN, maxN;

    f = this->colData.As<float>();
    for (size_t i = 0; i < all_cnt; i++) {
        int n = kdTree->annkFRSearch(dataPts[i], rad_sq, 0);
        if (i == 0) minN = maxN = n;
        else {
            if (n < minN) minN = n;
            if (n > maxN) maxN = n;
        }

        f[i * 4 + 3] = static_cast<float>(n);
    }

    if (autoScale) {
        this->densityMinCountSlot.Param<core::param::IntParam>()->SetValue(minN, false);
        this->densityMaxCountSlot.Param<core::param::IntParam>()->SetValue(maxN, false);
    } else {
        minN = this->densityMinCountSlot.Param<core::param::IntParam>()->Value();
        maxN = this->densityMaxCountSlot.Param<core::param::IntParam>()->Value();
    }

    for (size_t i = 0; i < all_cnt; i++) {
        f[i * 4 + 3] = (f[i * 4 + 3] - static_cast<float>(minN)) / static_cast<float>(maxN - minN);
        if (f[i * 4 + 3] < 0.0f) f[i * 4 + 3] = 0.0f;
        if (f[i * 4 + 3] > 1.0f) f[i * 4 + 3] = 1.0f;
    }

    delete kdTree;

    delete[] dataPts;
    delete[] dataPtsData;

    // for test purposes, map the opacity to color:
    const bool map_opacity_to_color = this->mapDensityToColorSlot.Param<core::param::BoolParam>()->Value();
    const bool remove_opacity = !this->mapDensityToAlphaSlot.Param<core::param::BoolParam>()->Value();
    if (map_opacity_to_color) {
        float *f = this->colData.As<float>();
        const vislib::graphics::ColourRGBAu8 c[5] = {
            vislib::graphics::ColourRGBAu8(0, 0, 255, 255),
            vislib::graphics::ColourRGBAu8(0, 255, 255, 255),
            vislib::graphics::ColourRGBAu8(0, 255, 0, 255),
            vislib::graphics::ColourRGBAu8(255, 255, 0, 255),
            vislib::graphics::ColourRGBAu8(255, 0, 0, 255) };
        for (size_t i = 0; i < all_cnt; i++) {

            float a = f[i * 4 + 3];

            if (remove_opacity) f[i * 4 + 3] = 1.0;

            a *= 4;
            vislib::graphics::ColourRGBAu8 pc;
            if (a >= 4) pc = c[4];
            else {
                int ci = static_cast<int>(a);
                pc = c[ci].Interpolate(c[ci + 1], a - static_cast<float>(ci));
            }

            f[i * 4 + 0] = static_cast<float>(pc.R()) / 255.0f;
            f[i * 4 + 1] = static_cast<float>(pc.G()) / 255.0f;
            f[i * 4 + 2] = static_cast<float>(pc.B()) / 255.0f;
        }
    } else if (remove_opacity) {
        for (size_t i = 0; i < all_cnt; i++) {
            f[i * 4 + 3] = 1.0;
        }
    }

}
