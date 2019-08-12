#include "stdafx.h"
#include "SpectralIntensityVolume.h"

#include <atomic>
#include <fstream>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "vislib/sys/ConsoleProgressBar.h"

#include "omp.h"

#include "simultaneous_sort.h"


megamol::astro::SpectralIntensityVolume::SpectralIntensityVolume()
    : volume_in_slot_("volumeIn", "Input of volume containing optical depth")
    , temp_in_slot_("tempIn", "Input of volume containing temperature")
    , mass_in_slot_("massIn", "Input of volume containing mass")
    , mw_in_slot_("mwIn", "Input of volume containing molecular weight")
    , astro_in_slot_("astroIn", "Input of astro particles")
    , volume_out_slot_("volumeOut", "Output of spectral intensity volume")
    , lsu_out_slot_("lsuOut", "Output of Bremsstrahlungs volume")
    , xResSlot("sizex", "The size of the volume in numbers of voxels")
    , yResSlot("sizey", "The size of the volume in numbers of voxels")
    , zResSlot("sizez", "The size of the volume in numbers of voxels")
    , cyclXSlot("cyclX", "Considers cyclic boundary conditions in X direction")
    , cyclYSlot("cyclY", "Considers cyclic boundary conditions in Y direction")
    , cyclZSlot("cyclZ", "Considers cyclic boundary conditions in Z direction")
    , normalizeSlot("normalize", "Normalize the output volume")
    , wavelength_slot_("wavelength", "Set the wavelength for the spectral intensity (in nm)") {
    volume_in_slot_.SetCompatibleCall<core::misc::VolumetricDataCallDescription>();
    MakeSlotAvailable(&volume_in_slot_);

    temp_in_slot_.SetCompatibleCall<core::misc::VolumetricDataCallDescription>();
    MakeSlotAvailable(&temp_in_slot_);

    mass_in_slot_.SetCompatibleCall<core::misc::VolumetricDataCallDescription>();
    MakeSlotAvailable(&mass_in_slot_);

    mw_in_slot_.SetCompatibleCall<core::misc::VolumetricDataCallDescription>();
    MakeSlotAvailable(&mw_in_slot_);

    astro_in_slot_.SetCompatibleCall<AstroDataCallDescription>();
    MakeSlotAvailable(&astro_in_slot_);

    this->volume_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_DATA),
        &SpectralIntensityVolume::getDataCallback);
    this->volume_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_EXTENTS),
        &SpectralIntensityVolume::getExtentCallback);
    this->volume_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_METADATA),
        &SpectralIntensityVolume::getExtentCallback);
    this->volume_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_START_ASYNC),
        &SpectralIntensityVolume::dummyCallback);
    this->volume_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_STOP_ASYNC),
        &SpectralIntensityVolume::dummyCallback);
    this->volume_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_TRY_GET_DATA),
        &SpectralIntensityVolume::dummyCallback);
    this->MakeSlotAvailable(&this->volume_out_slot_);

    this->lsu_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_DATA),
        &SpectralIntensityVolume::getLSUDataCallback);
    this->lsu_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_EXTENTS),
        &SpectralIntensityVolume::getExtentCallback);
    this->lsu_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_METADATA),
        &SpectralIntensityVolume::getExtentCallback);
    this->lsu_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_START_ASYNC),
        &SpectralIntensityVolume::dummyCallback);
    this->lsu_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_STOP_ASYNC),
        &SpectralIntensityVolume::dummyCallback);
    this->lsu_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_TRY_GET_DATA),
        &SpectralIntensityVolume::dummyCallback);
    this->MakeSlotAvailable(&this->lsu_out_slot_);

    this->xResSlot << new core::param::IntParam(16);
    this->MakeSlotAvailable(&this->xResSlot);
    this->yResSlot << new core::param::IntParam(16);
    this->MakeSlotAvailable(&this->yResSlot);
    this->zResSlot << new core::param::IntParam(16);
    this->MakeSlotAvailable(&this->zResSlot);

    this->cyclXSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclXSlot);
    this->cyclYSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclYSlot);
    this->cyclZSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclZSlot);

    this->normalizeSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->normalizeSlot);

    this->wavelength_slot_ << new core::param::FloatParam(1.0f, std::numeric_limits<float>::min(), 1000.f);
    MakeSlotAvailable(&wavelength_slot_);
}


megamol::astro::SpectralIntensityVolume::~SpectralIntensityVolume() { this->Release(); }


bool megamol::astro::SpectralIntensityVolume::create() { return true; }


void megamol::astro::SpectralIntensityVolume::release(void) {
    delete[] this->metadata.MinValues;
    delete[] this->metadata.MaxValues;
    delete[] this->metadata.SliceDists[0];
    delete[] this->metadata.SliceDists[1];
    delete[] this->metadata.SliceDists[2];
}


bool megamol::astro::SpectralIntensityVolume::getExtentCallback(core::Call& c) {
    auto* out = dynamic_cast<core::misc::VolumetricDataCall*>(&c);
    if (out == nullptr) return false;

    auto* ast = this->astro_in_slot_.CallAs<AstroDataCall>();
    if (ast == nullptr) return false;

    // if (!this->assertData(inMpdc, outDpdc)) return false;
    ast->SetFrameID(out->FrameID(), true);
    if (!(*ast)(1)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: could not get current frame extents (%u)", time - 1);
        return false;
    }
    out->AccessBoundingBoxes().SetObjectSpaceBBox(ast->GetBoundingBoxes().ObjectSpaceBBox());
    out->AccessBoundingBoxes().SetObjectSpaceClipBox(ast->GetBoundingBoxes().ObjectSpaceClipBox());
    out->SetFrameCount(ast->FrameCount());

    return true;
}


bool megamol::astro::SpectralIntensityVolume::getDataCallback(core::Call& c) {
    auto* ast = this->astro_in_slot_.CallAs<AstroDataCall>();
    if (ast == nullptr) return false;

    auto* vdc = this->volume_in_slot_.CallAs<core::misc::VolumetricDataCall>();
    if (vdc == nullptr) return false;

    auto* tdc = this->temp_in_slot_.CallAs<core::misc::VolumetricDataCall>();
    if (tdc == nullptr) return false;

    auto* mdc = this->mass_in_slot_.CallAs<core::misc::VolumetricDataCall>();
    if (mdc == nullptr) return false;

    auto* mwdc = this->mw_in_slot_.CallAs<core::misc::VolumetricDataCall>();
    if (mwdc == nullptr) return false;

    auto* outVol = dynamic_cast<core::misc::VolumetricDataCall*>(&c);
    if (outVol == nullptr) return false;

    ast->SetFrameID(outVol->FrameID(), true);
    vdc->SetFrameID(outVol->FrameID(), true);
    tdc->SetFrameID(outVol->FrameID(), true);
    mdc->SetFrameID(outVol->FrameID(), true);
    mwdc->SetFrameID(outVol->FrameID(), true);
    if (!(*ast)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get extents.");
        return false;
    }
    if (!(*ast)(0)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get data.");
        return false;
    }
    if (!(*vdc)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get volume metadata.");
        return false;
    }
    if (!(*vdc)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get volume extents.");
        return false;
    }
    if (!(*vdc)(core::misc::VolumetricDataCall::IDX_GET_DATA)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get volume data.");
        return false;
    }
    if (!(*tdc)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get temperature volume metadata.");
        return false;
    }
    if (!(*tdc)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get temperature volume extents.");
        return false;
    }
    if (!(*tdc)(core::misc::VolumetricDataCall::IDX_GET_DATA)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get temperature volume data.");
        return false;
    }
    if (!(*mdc)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get mass volume metadata.");
        return false;
    }
    if (!(*mdc)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get mass volume extents.");
        return false;
    }
    if (!(*mdc)(core::misc::VolumetricDataCall::IDX_GET_DATA)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get mass volume data.");
        return false;
    }

    if (!(*mwdc)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get molecular weight volume metadata.");
        return false;
    }
    if (!(*mwdc)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get molecular weight volume extents.");
        return false;
    }
    if (!(*mwdc)(core::misc::VolumetricDataCall::IDX_GET_DATA)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get molecular weight volume data.");
        return false;
    }
    if (this->time != ast->FrameID() || this->time != vdc->FrameID() || this->time != tdc->FrameID() ||
        this->time != mdc->FrameID() || this->time != mwdc->FrameID() || this->in_datahash != ast->DataHash() ||
        this->anythingDirty()) {
        if (!this->createVolumeCPU(*vdc, *tdc, *mdc, *mwdc, *ast)) return false;
        this->time = ast->FrameID();
        this->in_datahash = ast->DataHash();
        ++this->datahash;
        this->resetDirty();
    }

    // TODO set data
    outVol->SetData(this->vol_[0].data());
    metadata.Components = 1; //< TODO Maybe we want several wavelengths simultaneously
    metadata.GridType = core::misc::GridType_t::CARTESIAN;
    metadata.Resolution[0] = static_cast<size_t>(this->xResSlot.Param<core::param::IntParam>()->Value());
    metadata.Resolution[1] = static_cast<size_t>(this->yResSlot.Param<core::param::IntParam>()->Value());
    metadata.Resolution[2] = static_cast<size_t>(this->zResSlot.Param<core::param::IntParam>()->Value());
    metadata.ScalarType = core::misc::ScalarType_t::FLOATING_POINT;
    metadata.ScalarLength = sizeof(float);
    metadata.MinValues = new double[1];
    metadata.MinValues[0] = this->min_dens_;
    metadata.MaxValues = new double[1];
    metadata.MaxValues[0] = this->max_dens_;
    auto const bbox = ast->AccessBoundingBoxes().ObjectSpaceBBox();
    metadata.Extents[0] = bbox.Width();
    metadata.Extents[1] = bbox.Height();
    metadata.Extents[2] = bbox.Depth();
    metadata.NumberOfFrames = 1;
    metadata.SliceDists[0] = new float[1];
    metadata.SliceDists[0][0] = metadata.Extents[0] / static_cast<float>(metadata.Resolution[0] - 1);
    metadata.SliceDists[1] = new float[1];
    metadata.SliceDists[1][0] = metadata.Extents[1] / static_cast<float>(metadata.Resolution[1] - 1);
    metadata.SliceDists[2] = new float[1];
    metadata.SliceDists[2][0] = metadata.Extents[2] / static_cast<float>(metadata.Resolution[2] - 1);

    metadata.Origin[0] = bbox.Left();
    metadata.Origin[1] = bbox.Bottom();
    metadata.Origin[2] = bbox.Back();

    metadata.IsUniform[0] = true;
    metadata.IsUniform[1] = true;
    metadata.IsUniform[2] = true;
    outVol->SetMetadata(&metadata);

    outVol->SetDataHash(this->datahash);

    return true;
}


bool megamol::astro::SpectralIntensityVolume::getLSUDataCallback(core::Call& c) {
    auto* ast = this->astro_in_slot_.CallAs<AstroDataCall>();
    if (ast == nullptr) return false;

    auto* vdc = this->volume_in_slot_.CallAs<core::misc::VolumetricDataCall>();
    if (vdc == nullptr) return false;

    auto* tdc = this->temp_in_slot_.CallAs<core::misc::VolumetricDataCall>();
    if (tdc == nullptr) return false;

    auto* mdc = this->mass_in_slot_.CallAs<core::misc::VolumetricDataCall>();
    if (mdc == nullptr) return false;

    auto* mwdc = this->mw_in_slot_.CallAs<core::misc::VolumetricDataCall>();
    if (mwdc == nullptr) return false;

    auto* outVol = dynamic_cast<core::misc::VolumetricDataCall*>(&c);
    if (outVol == nullptr) return false;

    ast->SetFrameID(outVol->FrameID(), true);
    vdc->SetFrameID(outVol->FrameID(), true);
    tdc->SetFrameID(outVol->FrameID(), true);
    mdc->SetFrameID(outVol->FrameID(), true);
    mwdc->SetFrameID(outVol->FrameID(), true);
    if (!(*ast)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get extents.");
        return false;
    }
    if (!(*ast)(0)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get data.");
        return false;
    }
    if (!(*vdc)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get volume metadata.");
        return false;
    }
    if (!(*vdc)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get volume extents.");
        return false;
    }
    if (!(*vdc)(core::misc::VolumetricDataCall::IDX_GET_DATA)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get volume data.");
        return false;
    }
    if (!(*tdc)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get temperature volume metadata.");
        return false;
    }
    if (!(*tdc)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get temperature volume extents.");
        return false;
    }
    if (!(*tdc)(core::misc::VolumetricDataCall::IDX_GET_DATA)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get temperature volume data.");
        return false;
    }
    if (!(*mdc)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get mass volume metadata.");
        return false;
    }
    if (!(*mdc)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get mass volume extents.");
        return false;
    }
    if (!(*mdc)(core::misc::VolumetricDataCall::IDX_GET_DATA)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get mass volume data.");
        return false;
    }

    if (!(*mwdc)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get molecular weight volume metadata.");
        return false;
    }
    if (!(*mwdc)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get molecular weight volume extents.");
        return false;
    }
    if (!(*mwdc)(core::misc::VolumetricDataCall::IDX_GET_DATA)) {
        vislib::sys::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get molecular weight volume data.");
        return false;
    }
    if (this->time != ast->FrameID() || this->time != vdc->FrameID() || this->time != tdc->FrameID() ||
        this->time != mdc->FrameID() || this->time != mwdc->FrameID() || this->in_datahash != ast->DataHash() ||
        this->anythingDirty()) {
        if (!this->createBremsstrahlungVolume(*vdc, *tdc, *mdc, *mwdc, *ast)) return false;
        this->time = ast->FrameID();
        this->in_datahash = ast->DataHash();
        ++this->datahash;
        this->resetDirty();
    }

    // TODO set data
    outVol->SetData(this->vol_[0].data());
    metadata.Components = 1; //< TODO Maybe we want several wavelengths simultaneously
    metadata.GridType = core::misc::GridType_t::CARTESIAN;
    metadata.Resolution[0] = static_cast<size_t>(this->xResSlot.Param<core::param::IntParam>()->Value());
    metadata.Resolution[1] = static_cast<size_t>(this->yResSlot.Param<core::param::IntParam>()->Value());
    metadata.Resolution[2] = static_cast<size_t>(this->zResSlot.Param<core::param::IntParam>()->Value());
    metadata.ScalarType = core::misc::ScalarType_t::FLOATING_POINT;
    metadata.ScalarLength = sizeof(float);
    metadata.MinValues = new double[1];
    metadata.MinValues[0] = this->min_dens_;
    metadata.MaxValues = new double[1];
    metadata.MaxValues[0] = this->max_dens_;
    auto const bbox = ast->AccessBoundingBoxes().ObjectSpaceBBox();
    metadata.Extents[0] = bbox.Width();
    metadata.Extents[1] = bbox.Height();
    metadata.Extents[2] = bbox.Depth();
    metadata.NumberOfFrames = 1;
    metadata.SliceDists[0] = new float[1];
    metadata.SliceDists[0][0] = metadata.Extents[0] / static_cast<float>(metadata.Resolution[0] - 1);
    metadata.SliceDists[1] = new float[1];
    metadata.SliceDists[1][0] = metadata.Extents[1] / static_cast<float>(metadata.Resolution[1] - 1);
    metadata.SliceDists[2] = new float[1];
    metadata.SliceDists[2][0] = metadata.Extents[2] / static_cast<float>(metadata.Resolution[2] - 1);

    metadata.Origin[0] = bbox.Left();
    metadata.Origin[1] = bbox.Bottom();
    metadata.Origin[2] = bbox.Back();

    metadata.IsUniform[0] = true;
    metadata.IsUniform[1] = true;
    metadata.IsUniform[2] = true;
    outVol->SetMetadata(&metadata);

    outVol->SetDataHash(this->datahash);

    return true;
}


bool megamol::astro::SpectralIntensityVolume::createVolumeCPU(core::misc::VolumetricDataCall const& volumeIn,
    core::misc::VolumetricDataCall const& tempIn, core::misc::VolumetricDataCall const& massIn,
    core::misc::VolumetricDataCall const& mwIn, AstroDataCall& astroIn) {
    vislib::sys::Log::DefaultLog.WriteInfo("SpectralIntensityVolume: Starting volume creation.");

    auto const sx = this->xResSlot.Param<core::param::IntParam>()->Value();
    auto const sy = this->yResSlot.Param<core::param::IntParam>()->Value();
    auto const sz = this->zResSlot.Param<core::param::IntParam>()->Value();

    float const wl = this->wavelength_slot_.Param<core::param::FloatParam>()->Value() / 1000000000.0f;

    auto const numCells = sx * sy * sz;

    vol_.resize(omp_get_max_threads());
#pragma omp parallel for
    for (int init = 0; init < omp_get_max_threads(); ++init) {
        vol_[init].resize(numCells);
        std::fill(vol_[init].begin(), vol_[init].end(), 0.0f);
    }

    auto const cycl_x = this->cyclXSlot.Param<core::param::BoolParam>()->Value();
    auto const cycl_y = this->cyclYSlot.Param<core::param::BoolParam>()->Value();
    auto const cycl_z = this->cyclZSlot.Param<core::param::BoolParam>()->Value();

    auto const minOSx = astroIn.AccessBoundingBoxes().ObjectSpaceBBox().Left();
    auto const minOSy = astroIn.AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
    auto const minOSz = astroIn.AccessBoundingBoxes().ObjectSpaceBBox().Back();
    auto const rangeOSx = astroIn.AccessBoundingBoxes().ObjectSpaceBBox().Width();
    auto const rangeOSy = astroIn.AccessBoundingBoxes().ObjectSpaceBBox().Height();
    auto const rangeOSz = astroIn.AccessBoundingBoxes().ObjectSpaceBBox().Depth();
    auto const halfRangeOSx = 0.5f * rangeOSx;
    auto const halfRangeOSy = 0.5f * rangeOSy;
    auto const halfRangeOSz = 0.5f * rangeOSz;

    auto const sliceDistX = rangeOSx / static_cast<float>(sx - 1);
    auto const sliceDistY = rangeOSy / static_cast<float>(sy - 1);
    auto const sliceDistZ = rangeOSz / static_cast<float>(sz - 1);

    auto const max_spec_dist = std::max(sliceDistX, std::max(sliceDistY, sliceDistZ));
    // auto const cut_off = 10.0f * max_spec_dist;
    auto constexpr cut_off = std::numeric_limits<float>::max();


    auto positions = *astroIn.GetPositions().get();
    auto velo = *astroIn.GetVelocities().get();
    auto dens_f = *astroIn.GetDensity().get();
    auto sl = *astroIn.GetSmoothingLength().get();
    auto temps_f = *astroIn.GetTemperature().get();

    auto isBaryon = astroIn.GetIsBaryonFlags();
    std::vector<char> ib(isBaryon->size());
    for (size_t idx = 0; idx < isBaryon->size(); ++idx) {
        if (isBaryon->operator[](idx)) {
            ib[idx] = 1;
        } else {
            ib[idx] = 0;
        }
    }

    sort_with([](auto a, auto b) { return a > b; }, ib, positions, velo, dens_f, sl, temps_f);

    auto it = std::find(ib.cbegin(), ib.cend(), false);
    auto idx = std::distance(ib.cbegin(), it);

    positions.erase(positions.begin() + idx, positions.end());
    velo.erase(velo.begin() + idx, velo.end());
    dens_f.erase(dens_f.begin() + idx, dens_f.end());
    sl.erase(sl.begin() + idx, sl.end());
    temps_f.erase(temps_f.begin() + idx, temps_f.end());

    std::vector<double> dens(dens_f.size());
    std::copy(dens_f.cbegin(), dens_f.cend(), dens.begin());
    std::vector<double> temps(temps_f.size());
    std::copy(temps_f.cbegin(), temps_f.cend(), temps.begin());

    auto const minmax_dens = std::minmax_element(dens.begin(), dens.end());
    auto const min_dens = *minmax_dens.first;
    auto const minmax_dens_rcp = 1.0 / ((*minmax_dens.second) - min_dens);
    auto const minmax_temp = std::minmax_element(temps.begin(), temps.end());
    auto const min_temp = *minmax_temp.first;
    auto const minmax_temp_rcp = 1.0 / ((*minmax_temp.second) - min_temp);

    // normalize attributes
    std::transform(dens.cbegin(), dens.cend(), dens.begin(),
        [min_dens, minmax_dens_rcp](auto& val) { return (val - min_dens) * minmax_dens_rcp; });
    std::transform(temps.cbegin(), temps.cend(), temps.begin(),
        [min_temp, minmax_temp_rcp](auto& val) { return (val - min_temp) * minmax_temp_rcp; });

    std::vector<double> radiance(dens.size());
    std::transform(dens.cbegin(), dens.cend(), temps.cbegin(), radiance.begin(),
        [](double d, double t) { return d * d * std::sqrt(t); });
    std::transform(sl.cbegin(), sl.cend(), radiance.cbegin(), radiance.begin(),
        [](float r, double rad) { return 4.0 * 3.14 * rad * static_cast<double>(r * r * r); });


    // prepare input volume
    auto metadata = volumeIn.GetMetadata();
    auto volume = reinterpret_cast<float const*>(volumeIn.GetData());

    auto vol_min = metadata->MinValues[0];
    auto vol_max = metadata->MaxValues[0];

    auto vol_sx = metadata->Resolution[0];
    auto vol_sy = metadata->Resolution[1];
    auto vol_sz = metadata->Resolution[2];

    auto vol_disx = metadata->SliceDists[0][0];
    auto vol_disy = metadata->SliceDists[1][0];
    auto vol_disz = metadata->SliceDists[2][0];

    float min_vol_dis = std::min(vol_disx, std::min(vol_disy, vol_disz));

    auto vol_orgx = metadata->Origin[0];
    auto vol_orgy = metadata->Origin[1];
    auto vol_orgz = metadata->Origin[2];

    // prepare input temperature
    metadata = tempIn.GetMetadata();
    auto temperature = reinterpret_cast<float const*>(tempIn.GetData());

    auto temp_vol_min = metadata->MinValues[0];
    auto temp_vol_max = metadata->MaxValues[0];

    auto temp_vol_sx = metadata->Resolution[0];
    auto temp_vol_sy = metadata->Resolution[1];
    auto temp_vol_sz = metadata->Resolution[2];

    auto temp_vol_disx = metadata->SliceDists[0][0];
    auto temp_vol_disy = metadata->SliceDists[1][0];
    auto temp_vol_disz = metadata->SliceDists[2][0];

    float temp_min_vol_dis = std::min(temp_vol_disx, std::min(temp_vol_disy, temp_vol_disz));

    auto temp_vol_orgx = metadata->Origin[0];
    auto temp_vol_orgy = metadata->Origin[1];
    auto temp_vol_orgz = metadata->Origin[2];

    // prepare input mass
    metadata = massIn.GetMetadata();
    auto mass = reinterpret_cast<float const*>(massIn.GetData());

    auto mass_vol_min = metadata->MinValues[0];
    auto mass_vol_max = metadata->MaxValues[0];

    auto mass_vol_sx = metadata->Resolution[0];
    auto mass_vol_sy = metadata->Resolution[1];
    auto mass_vol_sz = metadata->Resolution[2];

    auto mass_vol_disx = metadata->SliceDists[0][0];
    auto mass_vol_disy = metadata->SliceDists[1][0];
    auto mass_vol_disz = metadata->SliceDists[2][0];

    float mass_min_vol_dis = std::min(mass_vol_disx, std::min(mass_vol_disy, mass_vol_disz));

    auto mass_vol_orgx = metadata->Origin[0];
    auto mass_vol_orgy = metadata->Origin[1];
    auto mass_vol_orgz = metadata->Origin[2];

    // prepare input molecular weight
    metadata = mwIn.GetMetadata();
    auto mw = reinterpret_cast<float const*>(mwIn.GetData());

    auto mw_vol_min = metadata->MinValues[0];
    auto mw_vol_max = metadata->MaxValues[0];

    auto mw_vol_sx = metadata->Resolution[0];
    auto mw_vol_sy = metadata->Resolution[1];
    auto mw_vol_sz = metadata->Resolution[2];

    auto mw_vol_disx = metadata->SliceDists[0][0];
    auto mw_vol_disy = metadata->SliceDists[1][0];
    auto mw_vol_disz = metadata->SliceDists[2][0];

    float mw_min_vol_dis = std::min(mw_vol_disx, std::min(mw_vol_disy, mw_vol_disz));

    auto mw_vol_orgx = metadata->Origin[0];
    auto mw_vol_orgy = metadata->Origin[1];
    auto mw_vol_orgz = metadata->Origin[2];

    if (sx != vol_sx || sx != temp_vol_sx || sx != mass_vol_sx || sx != mw_vol_sx || sy != vol_sy ||
        sy != temp_vol_sy || sy != mass_vol_sy || sy != mw_vol_sy || sz != vol_sz || sz != temp_vol_sz ||
        sz != mass_vol_sz || sz != mw_vol_sz) {
        vislib::sys::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Input volume size not compatible to requested output.");
        return false;
    }

    auto const cell_vol = vol_disx * vol_disy * vol_disz;
    //    auto const num_cells = vol_sx * vol_sy * vol_sz;

    std::vector<double> optical(numCells);
    std::transform(mw, mw + numCells, temperature, optical.begin(), [](float mw, float t) {
        return 0.018 * std::pow(static_cast<double>(t), -1.5) * 0.0134 * 0.0134 * static_cast<double>(mw) * 1.2;
    });
    std::transform(mass, mass + numCells, optical.cbegin(), optical.begin(), [](float m, double o) { return o / m; });
    auto const minmax_optical = std::minmax_element(optical.cbegin(), optical.cend());
    auto const min_optical = *minmax_optical.first;
    auto const minmax_optical_rcp = 1.0 / (*minmax_optical.second - min_optical);
    std::transform(optical.cbegin(), optical.cend(), optical.begin(),
        [min_optical, minmax_optical_rcp](float o) { return (o - min_optical) * minmax_optical_rcp; });

    // constexpr double kb = 1.380649e-23; // [J/K]
    // constexpr double hp = 6.626070e-34; // [J*s]
    // constexpr double c = 299792458.0;   // [m/s] (vacuum)
    // constexpr double c_pc = 0.307;      // [pc/y] (vacuum)

    // constexpr double pref = 2.0 * hp * c * c;
    // constexpr double suff = hp * c;

    // auto plank = [kb, hp, c, pref, suff](double wl, double t) -> double {
    //    auto const a0 = pref / std::pow(wl, 5);
    //    auto const b0 = std::exp(suff / (wl * kb * t)) - 1.0;
    //    return a0 / b0;
    //};

    // auto bremsstrahlung = [](double rad, double d, double t) -> double {
    //    // return d * d * std::sqrt(t) * 4.0 * 0.333 * 3.14 * rad * rad * rad;
    //    return d * d * std::sqrt(t) * 4.0 * 3.14 * rad * rad * rad;
    //};

    // auto absorption = [kb, hp](double d, double t) -> double {
    //    // if (hp * 10e17 < kb * t)
    //    // return 0.018 * std::pow(t, -1.5) * 0.0134 * 0.0134 * d;//*10e-34;
    //    return std::pow((t * 1e-9), -1.5) * d * d * 1e-20;
    //};

    vislib::sys::ConsoleProgressBar cpb;
    std::atomic<int> counter(0);

    cpb.Start("Volume Creation", positions.size());

#pragma omp parallel for
    for (int64_t idx = 0; idx < positions.size(); ++idx) {
        auto const pos = positions[idx];
        auto const x_base = pos.x;
        auto x = static_cast<int>((x_base - minOSx) / sliceDistX);
        auto const y_base = pos.y;
        auto y = static_cast<int>((y_base - minOSy) / sliceDistY);
        auto const z_base = pos.z;
        auto z = static_cast<int>((z_base - minOSz) / sliceDistZ);
        auto const rad = sl[idx];

        auto e = radiance[idx];

        // iterate over every voxel
        for (int vz = 0; vz < sz; ++vz) {
            for (int vy = 0; vy < sy; ++vy) {
                for (int vx = 0; vx < sx; ++vx) {
                    float vox_x = static_cast<float>(vx) * sliceDistX + minOSx;
                    float vox_y = static_cast<float>(vy) * sliceDistY + minOSy;
                    float vox_z = static_cast<float>(vz) * sliceDistZ + minOSz;

                    float x_diff = std::fabs(vox_x - x_base);
                    if (cycl_x && x_diff > halfRangeOSx) {
                        x_diff -= rangeOSx;
                        // auto tmp_hx = (vx + 2 * sx) % sx;
                        auto tmp_hx = vx;
                        if (x < vx) {
                            tmp_hx -= sx;
                        } else {
                            tmp_hx += sx;
                        }
                        vox_x = static_cast<float>(tmp_hx) * sliceDistX + minOSx;
                    }
                    float y_diff = std::fabs(vox_y - y_base);
                    if (cycl_y && y_diff > halfRangeOSy) {
                        y_diff -= rangeOSy;
                        // auto tmp_hy = (vy + 2 * sy) % sy;
                        auto tmp_hy = vy;
                        if (y < vy) {
                            tmp_hy -= sy;
                        } else {
                            tmp_hy += sy;
                        }
                        vox_y = static_cast<float>(tmp_hy) * sliceDistY + minOSy;
                    }
                    float z_diff = std::fabs(vox_z - z_base);
                    if (cycl_z && z_diff > halfRangeOSz) {
                        z_diff -= rangeOSz;
                        // auto tmp_hz = (vz + 2 * sz) % sz;
                        auto tmp_hz = vz;
                        if (z < vz) {
                            tmp_hz -= sz;
                        } else {
                            tmp_hz += sz;
                        }
                        vox_z = static_cast<float>(tmp_hz) * sliceDistZ + minOSz;
                    }

                    float const dis = std::sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);

                    if (dis >= rad && dis <= cut_off) {
                        // trace ray
                        double att = 0.0;

                        float t_step = min_vol_dis;

                        glm::vec3 org = {x_base, y_base, z_base};
                        glm::vec3 dest = {vox_x, vox_y, vox_z};
                        glm::vec3 dir = glm::normalize(dest - org);
                        float t = rad;
                        float t_max = dis;
                        while (t <= t_max && att < 1.0f) {
                            glm::vec3 curr = org + t * dir;

                            auto ax = static_cast<int>((curr.x - vol_orgx) / vol_disx);
                            auto ay = static_cast<int>((curr.y - vol_orgy) / vol_disy);
                            auto az = static_cast<int>((curr.z - vol_orgz) / vol_disz);

                            if (ax < 0) ax += vol_sx;
                            if (ay < 0) ay += vol_sy;
                            if (az < 0) az += vol_sz;
                            if (ax >= vol_sx) ax -= vol_sx;
                            if (ay >= vol_sy) ay -= vol_sy;
                            if (az >= vol_sz) az -= vol_sz;

                            att += optical[(az * vol_sy + ay) * vol_sx + ax];

                            t += t_step;
                        }

                        vol_[omp_get_thread_num()][(vz * sy + vy) * sx + vx] += e * (1.0 - att);
                    }
                }
            }
        }

        ++counter;
        if (omp_get_thread_num() == 0) {
            cpb.Set(counter.load());
        }
    }
    cpb.Stop();

    for (int i = 1; i < omp_get_max_threads(); ++i) {
        std::transform(vol_[i].begin(), vol_[i].end(), vol_[0].begin(), vol_[0].begin(), std::plus<>());
    }

    max_dens_ = *std::max_element(vol_[0].begin(), vol_[0].end());
    min_dens_ = *std::min_element(vol_[0].begin(), vol_[0].end());
    vislib::sys::Log::DefaultLog.WriteInfo(
        "SpectralIntensityVolume: Captured intensity %f -> %f", min_dens_, max_dens_);

    if (this->normalizeSlot.Param<core::param::BoolParam>()->Value()) {
        auto const rcpValRange = 1.0f / (max_dens_ - min_dens_);
        std::transform(vol_[0].begin(), vol_[0].end(), vol_[0].begin(),
            [this, rcpValRange](float const& a) { return (a - min_dens_) * rcpValRange; });
        min_dens_ = 0.0f;
        max_dens_ = 1.0f;
    }

//#define SIV_DEBUG_OUTPUT
#ifdef SIV_DEBUG_OUTPUT
    std::ofstream raw_file{"int.raw", std::ios::binary};
    raw_file.write(reinterpret_cast<char const*>(vol_[0].data()), vol_[0].size() * sizeof(float));
    raw_file.close();
    vislib::sys::Log::DefaultLog.WriteInfo("SpectralIntensityVolume: Debug file written\n");
#endif

    // Cleanup
    vol_.resize(1);

    return true;
}


bool megamol::astro::SpectralIntensityVolume::createBremsstrahlungVolume(core::misc::VolumetricDataCall const& volumeIn,
    core::misc::VolumetricDataCall const& tempIn, core::misc::VolumetricDataCall const& massIn,
    core::misc::VolumetricDataCall const& mwIn, AstroDataCall& astroIn) {
    vislib::sys::Log::DefaultLog.WriteInfo("SpectralIntensityVolume: Starting volume creation.");

    sx = this->xResSlot.Param<core::param::IntParam>()->Value();
    sy = this->yResSlot.Param<core::param::IntParam>()->Value();
    sz = this->zResSlot.Param<core::param::IntParam>()->Value();

    auto numCells = sx * sy * sz;


    // prepare input volume
    auto metadata = volumeIn.GetMetadata();
    auto density = reinterpret_cast<float const*>(volumeIn.GetData());

    auto vol_min = metadata->MinValues[0];
    auto vol_max = metadata->MaxValues[0];

    auto vol_sx = metadata->Resolution[0];
    auto vol_sy = metadata->Resolution[1];
    auto vol_sz = metadata->Resolution[2];

    auto vol_disx = metadata->SliceDists[0][0];
    auto vol_disy = metadata->SliceDists[1][0];
    auto vol_disz = metadata->SliceDists[2][0];

    // prepare input temperature
    metadata = tempIn.GetMetadata();
    auto temperature = reinterpret_cast<float const*>(tempIn.GetData());

    auto temp_vol_min = metadata->MinValues[0];
    auto temp_vol_max = metadata->MaxValues[0];

    auto temp_vol_sx = metadata->Resolution[0];
    auto temp_vol_sy = metadata->Resolution[1];
    auto temp_vol_sz = metadata->Resolution[2];


    if (vol_sx != temp_vol_sx || vol_sy != temp_vol_sy || vol_sz != temp_vol_sz) {
        vislib::sys::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Density and temperature volume are not compatible. Aborting.");
        return false;
    }

    if (vol_sx != sx || vol_sy != sy || vol_sz != sz) {
        vislib::sys::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Selected resolution not compatible to input. Resetting to %d %d %d.", vol_sx,
            vol_sy, vol_sz);
        sx = vol_sx;
        sy = vol_sy;
        sz = vol_sz;
    }

    numCells = vol_sx * vol_sy * vol_sz;

    auto const cell_vol = vol_disx * vol_disy * vol_disz;
    vol_.resize(1);
    vol_[0].resize(numCells);
    std::transform(density, density + numCells, temperature, vol_[0].begin(),
        [cell_vol](float d, float t) { return d * d * std::sqrt(t) * cell_vol; });

    max_dens_ = *std::max_element(vol_[0].begin(), vol_[0].end());
    min_dens_ = *std::min_element(vol_[0].begin(), vol_[0].end());
    vislib::sys::Log::DefaultLog.WriteInfo(
        "SpectralIntensityVolume: Captured intensity %f -> %f", min_dens_, max_dens_);

    if (this->normalizeSlot.Param<core::param::BoolParam>()->Value()) {
        auto const rcpValRange = 1.0f / (max_dens_ - min_dens_);
        std::transform(vol_[0].begin(), vol_[0].end(), vol_[0].begin(),
            [this, rcpValRange](float const& a) { return (a - min_dens_) * rcpValRange; });
        min_dens_ = 0.0f;
        max_dens_ = 1.0f;
    }

    return true;
}
