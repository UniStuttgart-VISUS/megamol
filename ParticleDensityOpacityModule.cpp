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
#include "param/EnumParam.h"


megamol::stdplugin::datatools::ParticleDensityOpacityModule::ParticleDensityOpacityModule(void) : Module(),
        putDataSlot("putdata", "Connects from the data consumer"),
        getDataSlot("getdata", "Connects to the data source"),
        rebuildButtonSlot("rebuild", "Forces rebuild of colour data"),
        densityRadiusSlot("density::radius", "The radius of the local volume for the density computation"),
        densityMinCountSlot("density::min", "The minimum density particle count"),
        densityMaxCountSlot("density::max", "The maximum density particle count"),
        densityComputeCountRangeSlot("density::computeRange", "Compute 'min' and 'max' values"),
        opacityMinValSlot("opacity::min", "Minimum opacity value"),
        opacityMaxValSlot("opacity::max", "Maximum opacity value"),
        cyclBoundXSlot("periodicBoundary::x", "Dis-/Enables periodic boundary conditions in x direction"),
        cyclBoundYSlot("periodicBoundary::y", "Dis-/Enables periodic boundary conditions in y direction"),
        cyclBoundZSlot("periodicBoundary::z", "Dis-/Enables periodic boundary conditions in z direction"),
        mapModeSlot("opacity::mapMode", "Mode to map the density to the data"),
        lastFrame(0), lastHash(0), colData(),
        densitAlgorithmSlot("density::algorithm", "The density computation algorithm to use"),
        tfQuery(),
        densityAutoComputeCountRangeSlot("density::autoComputeRange", "Automatically compute 'min' and 'max'") {

    this->putDataSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(), "GetData", &ParticleDensityOpacityModule::getDataCallback);
    this->putDataSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(), "GetExtent", &ParticleDensityOpacityModule::getExtentCallback);
    this->MakeSlotAvailable(&this->putDataSlot);

    this->getDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->MakeSlotAvailable(this->tfQuery.GetSlot());

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

    core::param::EnumParam *mapModeParam = new core::param::EnumParam(static_cast<int>(MapMode::Luminance));
    mapModeParam->SetTypePair(static_cast<int>(MapMode::AlphaOverwrite), "AlphaOverwrite");
    mapModeParam->SetTypePair(static_cast<int>(MapMode::ColorRainbow), "ColorRainbow");
    mapModeParam->SetTypePair(static_cast<int>(MapMode::ColorRainbowAlpha), "ColorRainbowAlpha");
    mapModeParam->SetTypePair(static_cast<int>(MapMode::Luminance), "Luminance");
    this->mapModeSlot.SetParameter(mapModeParam);
    this->MakeSlotAvailable(&this->mapModeSlot);

    core::param::EnumParam *dAlg = new core::param::EnumParam(static_cast<int>(DensityAlgorithmType::grid));
    dAlg->SetTypePair(static_cast<int>(DensityAlgorithmType::ANN), "ANN");
    dAlg->SetTypePair(static_cast<int>(DensityAlgorithmType::grid), "grid");
    this->densitAlgorithmSlot.SetParameter(dAlg);
    this->MakeSlotAvailable(&this->densitAlgorithmSlot);

    this->densityAutoComputeCountRangeSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->densityAutoComputeCountRangeSlot);
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
        if (this->densitAlgorithmSlot.IsDirty()) {
            this->densitAlgorithmSlot.ResetDirty();
            update_data = true;
        }
        if (this->densityComputeCountRangeSlot.IsDirty()) {
            update_data = true;
        }
        if (this->densityAutoComputeCountRangeSlot.IsDirty()) {
            this->densityAutoComputeCountRangeSlot.ResetDirty();
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
        if (this->mapModeSlot.IsDirty()) {
            this->mapModeSlot.ResetDirty();
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
            switch (static_cast<MapMode>(this->mapModeSlot.Param<core::param::EnumParam>()->Value())) {
            case MapMode::AlphaOverwrite: // fall through
            case MapMode::ColorRainbowAlpha:
                p.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA,
                    this->colData.At(cnt * sizeof(float) * 4));
                break;
            case MapMode::ColorRainbow:
                p.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGB,
                    this->colData.At(cnt * sizeof(float) * 4), sizeof(float) * 4);
                break;
            case MapMode::Luminance:
                p.SetColourMapIndexValues(0.0f, 1.0f);
                p.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I,
                    this->colData.At(cnt * sizeof(float)));
                break;
            }

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
    bool autoScale = this->densityComputeCountRangeSlot.IsDirty()
        || this->densityAutoComputeCountRangeSlot.Param<core::param::BoolParam>()->Value();

    size_t all_cnt = 0;
    unsigned int plc = dat->GetParticleListCount();
    for (unsigned int pli = 0; pli < plc; pli++) {
        core::moldyn::SimpleSphericalParticles::VertexDataType vdt = dat->AccessParticles(pli).GetVertexDataType();
        if ((vdt == core::moldyn::SimpleSphericalParticles::VERTDATA_NONE)
            || (vdt == core::moldyn::SimpleSphericalParticles::VERTDATA_SHORT_XYZ)) continue;
        all_cnt += static_cast<size_t>(dat->AccessParticles(pli).GetCount());
    }

    MapMode mmode = static_cast<MapMode>(this->mapModeSlot.Param<core::param::EnumParam>()->Value());
    bool use_rgba = (mmode != MapMode::Luminance);
    int col_step = (use_rgba ? 4 : 1);
    int col_off = (use_rgba ? 3 : 0);
    this->colData.EnforceSize(all_cnt * sizeof(float) * col_step);

    size_t ci = 0;
    float *f = this->colData.As<float>();
    if (use_rgba && (mmode != MapMode::ColorRainbowAlpha)) {
        // copy color values
        this->tfQuery.Clear();
        for (unsigned int pli = 0; pli < plc; pli++) {
            core::moldyn::MultiParticleDataCall::Particles &pl = dat->AccessParticles(pli);
            if ((pl.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_NONE)
                || (pl.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_SHORT_XYZ)) continue;
            size_t col_stride = 0;
            bool bytes = true;
            bool alpha = false;
            const uint8_t *pld = static_cast<const uint8_t*>(pl.GetColourData());

            switch (pl.GetColourDataType()) {
            case core::moldyn::SimpleSphericalParticles::COLDATA_NONE: { //< use global colour
                float r = static_cast<float>(pl.GetGlobalColour()[0]) / 255.0f;
                float g = static_cast<float>(pl.GetGlobalColour()[1]) / 255.0f;
                float b = static_cast<float>(pl.GetGlobalColour()[2]) / 255.0f;
                float a = static_cast<float>(pl.GetGlobalColour()[3]) / 255.0f;
                for (uint64_t pi = 0; pi < pl.GetCount(); pi++, ci++) {
                    f[ci * 4 + 0] = r;
                    f[ci * 4 + 1] = g;
                    f[ci * 4 + 2] = b;
                    f[ci * 4 + 3] = a;
                }
            } continue;
            case core::moldyn::SimpleSphericalParticles::COLDATA_UINT8_RGB: bytes = true; col_stride = 3; break;
            case core::moldyn::SimpleSphericalParticles::COLDATA_UINT8_RGBA: bytes = true; alpha = true; col_stride = 4; break;
            case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGB: bytes = false; col_stride = 12; break;
            case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA: bytes = false; alpha = true; col_stride = 16; break;
            case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I: { //< single float value to be mapped by a transfer function
                col_stride = (4 < pl.GetColourDataStride()) ? pl.GetColourDataStride() : 4;
                float cvmin = pl.GetMinColourIndexValue();
                float cvrng = pl.GetMaxColourIndexValue() - cvmin;
                if (cvrng <= 0.0f) cvrng = 0.0f; else cvrng = 1.0f / cvrng;
                for (uint64_t pi = 0; pi < pl.GetCount(); pi++, ci++, pld += col_stride) {
                    const float *plf = reinterpret_cast<const float*>(pld);
                    float c = (*plf - cvmin) * cvrng; // color value in [0..1]
                    this->tfQuery.Query(f + (ci * 4), c);
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
                    if (alpha) {
                        f[ci * 4 + 3] = static_cast<float>(plb[3]) / 255.0f;
                    }
                }
            } else {
                for (uint64_t pi = 0; pi < pl.GetCount(); pi++, ci++, pld += col_stride) {
                    const float *plf = reinterpret_cast<const float*>(pld);
                    f[ci * 4 + 0] = plf[0];
                    f[ci * 4 + 1] = plf[1];
                    f[ci * 4 + 2] = plf[2];
                    if (alpha) {
                        f[ci * 4 + 3] = plf[3];
                    }
                }
            }
        }
    }

    int minN, maxN;
    if (!autoScale) {
        minN = this->densityMinCountSlot.Param<core::param::IntParam>()->Value();
        maxN = this->densityMaxCountSlot.Param<core::param::IntParam>()->Value();
        if (maxN < minN) ::std::swap(minN, maxN);
        if (maxN == minN) maxN++;
    }

    // now copy position and compute density
    DensityAlgorithmType dAlg = static_cast<DensityAlgorithmType>(
        this->densitAlgorithmSlot.Param<core::param::EnumParam>()->Value());
    float rad = this->densityRadiusSlot.Param<core::param::FloatParam>()->Value();
    if (dAlg == DensityAlgorithmType::ANN) {
        // implementation using ANN
        ANNpointArray dataPts = new ANNpoint[all_cnt];
        ANNpoint dataPtsData = new ANNcoord[3 * all_cnt];

        // interate over all particles and store positions as ANN points
        for (unsigned int pli = 0; pli < plc; pli++) {
            core::moldyn::MultiParticleDataCall::Particles &pl = dat->AccessParticles(pli);
            if ((pl.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_NONE)
                || (pl.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_SHORT_XYZ)) continue;
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
        }

        ANNkd_tree* kdTree = new ANNkd_tree(dataPts, static_cast<int>(all_cnt), 3);
        ANNdist rad_sq = static_cast<ANNdist>(rad);
        rad_sq *= rad_sq;

        f = this->colData.As<float>();
        for (size_t i = 0; i < all_cnt; i++) {
            int n = kdTree->annkFRSearch(dataPts[i], rad_sq, 0);
            f[i * col_step + col_off] = static_cast<float>(n);
        }

        delete kdTree;

        delete[] dataPts;
        delete[] dataPtsData;

    } else {
        // use simple grid based implementation
        //double rad_sq = static_cast<double>(rad) * static_cast<double>(rad);
        float rad_sq = rad * rad;

        // count neighbours within 'rad'
        const vislib::math::Cuboid<float> &box = dat->AccessBoundingBoxes().ObjectSpaceClipBox();
        unsigned int dim_x = static_cast<unsigned int>(::std::ceil(box.Width() / rad));
        unsigned int dim_y = static_cast<unsigned int>(::std::ceil(box.Height() / rad));
        unsigned int dim_z = static_cast<unsigned int>(::std::ceil(box.Depth() / rad));

        unsigned int *cnt_grid = new unsigned int[dim_x * dim_y * dim_z];
        ::memset(cnt_grid, 0, sizeof(unsigned int) * dim_x * dim_y * dim_z);

        // Access coords[cell][particle][xyz]
        // coords[0] point to the whole particles access array
        float const ***coords = new float const **[dim_x * dim_y * dim_z];
        coords[0] = new float const *[all_cnt];

        // 1. count all particles for each cell
        ci = 0;
        for (unsigned int pli = 0; pli < plc; pli++) {
            core::moldyn::MultiParticleDataCall::Particles &pl = dat->AccessParticles(pli);
            if ((pl.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_NONE)
                || (pl.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_SHORT_XYZ)) continue;
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
                const float *v = reinterpret_cast<const float*>(vert);
                unsigned int x = static_cast<unsigned int>(::vislib::math::Clamp<float>((v[0] - box.Left()) / rad, 0.0f, static_cast<float>(dim_x - 1)));
                unsigned int y = static_cast<unsigned int>(::vislib::math::Clamp<float>((v[1] - box.Bottom()) / rad, 0.0f, static_cast<float>(dim_y - 1)));
                unsigned int z = static_cast<unsigned int>(::vislib::math::Clamp<float>((v[2] - box.Back()) / rad, 0.0f, static_cast<float>(dim_z - 1)));

                cnt_grid[x + dim_x * (y + dim_y * z)]++;
            }
        }

        // 2. allocate all coord-reference arrays for each cell
        ci = cnt_grid[0];
        for (unsigned int cell_idx = 1; cell_idx < dim_x * dim_y * dim_z; cell_idx++) {
            coords[cell_idx] = coords[0] + ci;
//            printf(" %u", cnt_grid[cell_idx]);
            ci += cnt_grid[cell_idx];
        }
//        printf("\n");
        assert(ci == all_cnt);

        // 3. place all particle coord references in each cell array
        ::memset(cnt_grid, 0, sizeof(unsigned int) * dim_x * dim_y * dim_z);
        ci = 0;
        for (unsigned int pli = 0; pli < plc; pli++) {
            core::moldyn::MultiParticleDataCall::Particles &pl = dat->AccessParticles(pli);
            if ((pl.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_NONE)
                || (pl.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_SHORT_XYZ)) continue;
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
                const float *v = reinterpret_cast<const float*>(vert);
                unsigned int x = static_cast<unsigned int>(::vislib::math::Clamp<float>((v[0] - box.Left()) / rad, 0.0f, static_cast<float>(dim_x - 1)));
                unsigned int y = static_cast<unsigned int>(::vislib::math::Clamp<float>((v[1] - box.Bottom()) / rad, 0.0f, static_cast<float>(dim_y - 1)));
                unsigned int z = static_cast<unsigned int>(::vislib::math::Clamp<float>((v[2] - box.Back()) / rad, 0.0f, static_cast<float>(dim_z - 1)));
                unsigned int cell_idx = x + dim_x * (y + dim_y * z);
                coords[cell_idx][cnt_grid[cell_idx]++] = v;
            }
        }

        // 4. finally compute distance information
        ci = 0;
        bool cyclX = this->cyclBoundXSlot.Param<core::param::BoolParam>()->Value();
        bool cyclY = this->cyclBoundYSlot.Param<core::param::BoolParam>()->Value();
        bool cyclZ = this->cyclBoundZSlot.Param<core::param::BoolParam>()->Value();
        for (unsigned int pli = 0; pli < plc; pli++) {
            core::moldyn::MultiParticleDataCall::Particles &pl = dat->AccessParticles(pli);
            if ((pl.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_NONE)
                || (pl.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_SHORT_XYZ)) continue;
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

            const int pi_cnt = static_cast<int>(pl.GetCount());
            #pragma omp parallel for
            for (int pi = 0; pi < pi_cnt; pi++) {
//            for (uint64_t pi = 0; pi < pl.GetCount(); pi++, vert += vert_stride) {
                size_t idx = static_cast<size_t>(pi + ci);
                const float *v = reinterpret_cast<const float*>(vert + vert_stride * pi);
                unsigned int x = static_cast<unsigned int>(::vislib::math::Clamp<float>((v[0] - box.Left()) / rad, 0.0f, static_cast<float>(dim_x - 1)));
                unsigned int y = static_cast<unsigned int>(::vislib::math::Clamp<float>((v[1] - box.Bottom()) / rad, 0.0f, static_cast<float>(dim_y - 1)));
                unsigned int z = static_cast<unsigned int>(::vislib::math::Clamp<float>((v[2] - box.Back()) / rad, 0.0f, static_cast<float>(dim_z - 1)));

                int n_cnt = 0;

                for (int ix = -1; ix <= 1; ix++) {
                    int cx = static_cast<int>(x) + ix;
                    if (cx == -1) {
                        if (cyclX) cx = dim_x - 1;
                        else continue;
                    } else if (cx == dim_x) {
                        if (cyclX) cx = 0;
                        else continue;
                    }
                    for (int iy = -1; iy <= 1; iy++) {
                        int cy = static_cast<int>(y) + iy;
                        if (cy == -1) {
                            if (cyclY) cy = dim_y - 1;
                            else continue;
                        } else if (cy == dim_y) {
                            if (cyclY) cy = 0;
                            else continue;
                        }
                        for (int iz = -1; iz <= 1; iz++) {
                            int cz = static_cast<int>(z) + iz;
                            if (cz == -1) {
                                if (cyclZ) cz = dim_z - 1;
                                else continue;
                            } else if (cz == dim_z) {
                                if (cyclZ) cz = 0;
                                else continue;
                            }

                            unsigned int c_idx = static_cast<unsigned int>(cx + dim_x * (cy + dim_y * cz));
                            for (int cpi = cnt_grid[c_idx] - 1; cpi >= 0; cpi--) {
                                const float *pf = coords[c_idx][cpi];
                                float dx = pf[0] - v[0];
                                float dy = pf[1] - v[1];
                                float dz = pf[2] - v[2];
                                dx *= dx;
                                dy *= dy;
                                dz *= dz;
                                float sqd = dx + dy + dz;
                                //double sqd = (static_cast<double>(pf[0] - v[0]) * static_cast<double>(pf[0] - v[0]))
                                //           + (static_cast<double>(pf[1] - v[1]) * static_cast<double>(pf[1] - v[1]))
                                //           + (static_cast<double>(pf[2] - v[2]) * static_cast<double>(pf[2] - v[2]));
                                if (sqd < rad_sq) {
                                    n_cnt++;
                                }
                            }
                        }
                    }
                }

                f[idx * col_step + col_off] = static_cast<float>(n_cnt);
            }
        }

        delete[] coords[0];
        delete[] coords;
        delete[] cnt_grid;

    }

    if (autoScale) {
        for (size_t i = 0; i < all_cnt; i++) {
            int n = static_cast<int>(f[i * col_step + col_off]);
            if (i == 0) minN = maxN = n;
            else {
                if (n < minN) minN = n;
                if (n > maxN) maxN = n;
            }
        }
        if (maxN < minN) ::std::swap(minN, maxN);
        if (maxN == minN) maxN++;
        this->densityMinCountSlot.Param<core::param::IntParam>()->SetValue(minN, false);
        this->densityMaxCountSlot.Param<core::param::IntParam>()->SetValue(maxN, false);
    }
    // printf("\n\tValue Range[%d, %d]\n\n", minN, maxN);

    for (size_t i = 0; i < all_cnt; i++) {
        f[i * col_step + col_off] = (f[i * col_step + col_off] - static_cast<float>(minN)) / static_cast<float>(maxN - minN);
        if (f[i * col_step + col_off] < 0.0f) f[i * col_step + col_off] = 0.0f;
        if (f[i * col_step + col_off] > 1.0f) f[i * col_step + col_off] = 1.0f;
    }

    // for test purposes, map the opacity to color:
    const bool map_opacity_to_color = ((mmode == MapMode::ColorRainbow) || (mmode == MapMode::ColorRainbowAlpha));
    const bool remove_opacity = (mmode == MapMode::ColorRainbow);
    if (map_opacity_to_color) {
        float *f = this->colData.As<float>();
        const vislib::graphics::ColourRGBAu8 c[5] = {
            vislib::graphics::ColourRGBAu8(0, 0, 255, 255),
            vislib::graphics::ColourRGBAu8(0, 255, 255, 255),
            vislib::graphics::ColourRGBAu8(0, 255, 0, 255),
            vislib::graphics::ColourRGBAu8(255, 255, 0, 255),
            vislib::graphics::ColourRGBAu8(255, 0, 0, 255) };
        for (size_t i = 0; i < all_cnt; i++) {

            float a = f[i * col_step + col_off];

            if (remove_opacity) f[i * col_step + col_off] = 1.0;

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
            f[i * col_step + col_off] = 1.0;
        }
    }

}
