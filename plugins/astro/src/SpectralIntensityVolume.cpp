#include "SpectralIntensityVolume.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <atomic>
#include <fstream>
#include <random>

#include <omp.h>
#include <simultaneous_sort/simultaneous_sort.h>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "vislib/sys/ConsoleProgressBar.h"

megamol::astro::SpectralIntensityVolume::SpectralIntensityVolume()
        : volume_in_slot_("volumeIn", "Input of volume containing optical depth")
        , temp_in_slot_("tempIn", "Input of volume containing temperature")
        , mass_in_slot_("massIn", "Input of volume containing mass")
        , mw_in_slot_("mwIn", "Input of volume containing molecular weight")
        , astro_in_slot_("astroIn", "Input of astro particles")
        , volume_out_slot_("volumeOut", "Output of spectral intensity volume")
        , lsu_out_slot_("lsuOut", "Output of Bremsstrahlungs volume")
        , absorption_out_slot_("absorptionOut", "Output of Absorption volume")
        , xResSlot("sizex", "The size of the volume in numbers of voxels")
        , yResSlot("sizey", "The size of the volume in numbers of voxels")
        , zResSlot("sizez", "The size of the volume in numbers of voxels")
        , cyclXSlot("cyclX", "Considers cyclic boundary conditions in X direction")
        , cyclYSlot("cyclY", "Considers cyclic boundary conditions in Y direction")
        , cyclZSlot("cyclZ", "Considers cyclic boundary conditions in Z direction")
        , normalizeSlot("normalize", "Normalize the output volume")
        //, wavelength_slot_("wavelength", "Set the wavelength for the spectral intensity (in nm)")
        , numSamplesSlot("numSamples", "Number of samples per particle in the darth volume case")
        , absorptionBiasSlot(
              "absorptionBias", "Determines influence of absorption coefficient in the darth volume case")
        , coneSampleNumSlot("coneNumSamples", "Number of samples for cone tracing in darth volume case")
        , coneAngleSlot("coneAngle", "Angle of the cone in the darth volume case (degree)") {
    volume_in_slot_.SetCompatibleCall<geocalls::VolumetricDataCallDescription>();
    MakeSlotAvailable(&volume_in_slot_);

    temp_in_slot_.SetCompatibleCall<geocalls::VolumetricDataCallDescription>();
    MakeSlotAvailable(&temp_in_slot_);

    mass_in_slot_.SetCompatibleCall<geocalls::VolumetricDataCallDescription>();
    MakeSlotAvailable(&mass_in_slot_);

    mw_in_slot_.SetCompatibleCall<geocalls::VolumetricDataCallDescription>();
    MakeSlotAvailable(&mw_in_slot_);

    astro_in_slot_.SetCompatibleCall<AstroDataCallDescription>();
    MakeSlotAvailable(&astro_in_slot_);

    this->volume_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_GET_DATA),
        &SpectralIntensityVolume::getDataCallback);
    this->volume_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_GET_EXTENTS),
        &SpectralIntensityVolume::getExtentCallback);
    this->volume_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_GET_METADATA),
        &SpectralIntensityVolume::getExtentCallback);
    this->volume_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_START_ASYNC),
        &SpectralIntensityVolume::dummyCallback);
    this->volume_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_STOP_ASYNC),
        &SpectralIntensityVolume::dummyCallback);
    this->volume_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_TRY_GET_DATA),
        &SpectralIntensityVolume::dummyCallback);
    this->MakeSlotAvailable(&this->volume_out_slot_);

    this->lsu_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_GET_DATA),
        &SpectralIntensityVolume::getLSUDataCallback);
    this->lsu_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_GET_EXTENTS),
        &SpectralIntensityVolume::getExtentCallback);
    this->lsu_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_GET_METADATA),
        &SpectralIntensityVolume::getExtentCallback);
    this->lsu_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_START_ASYNC),
        &SpectralIntensityVolume::dummyCallback);
    this->lsu_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_STOP_ASYNC),
        &SpectralIntensityVolume::dummyCallback);
    this->lsu_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_TRY_GET_DATA),
        &SpectralIntensityVolume::dummyCallback);
    this->MakeSlotAvailable(&this->lsu_out_slot_);

    this->absorption_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_GET_DATA),
        &SpectralIntensityVolume::getAbsorptionDataCallback);
    this->absorption_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_GET_EXTENTS),
        &SpectralIntensityVolume::getExtentCallback);
    this->absorption_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_GET_METADATA),
        &SpectralIntensityVolume::getExtentCallback);
    this->absorption_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_START_ASYNC),
        &SpectralIntensityVolume::dummyCallback);
    this->absorption_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_STOP_ASYNC),
        &SpectralIntensityVolume::dummyCallback);
    this->absorption_out_slot_.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_TRY_GET_DATA),
        &SpectralIntensityVolume::dummyCallback);
    this->MakeSlotAvailable(&this->absorption_out_slot_);

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

    /*this->wavelength_slot_ << new core::param::FloatParam(1.0f, std::numeric_limits<float>::min(), 1000.f);
    MakeSlotAvailable(&wavelength_slot_);*/

    numSamplesSlot << new core::param::IntParam(256, 1);
    MakeSlotAvailable(&numSamplesSlot);

    absorptionBiasSlot << new core::param::FloatParam(1.0f, -1.0f, 1.0f);
    MakeSlotAvailable(&absorptionBiasSlot);

    coneSampleNumSlot << new core::param::IntParam(4, 1);
    MakeSlotAvailable(&coneSampleNumSlot);

    coneAngleSlot << new core::param::FloatParam(2.0f, 0.001f, 90.0f);
    MakeSlotAvailable(&coneAngleSlot);
}


megamol::astro::SpectralIntensityVolume::~SpectralIntensityVolume() {
    this->Release();
}


bool megamol::astro::SpectralIntensityVolume::create() {
    return true;
}


void megamol::astro::SpectralIntensityVolume::release() {
    delete[] this->metadata.MinValues;
    delete[] this->metadata.MaxValues;
    delete[] this->metadata.SliceDists[0];
    delete[] this->metadata.SliceDists[1];
    delete[] this->metadata.SliceDists[2];
}


bool megamol::astro::SpectralIntensityVolume::getExtentCallback(core::Call& c) {
    auto* out = dynamic_cast<geocalls::VolumetricDataCall*>(&c);
    if (out == nullptr)
        return false;

    auto* ast = this->astro_in_slot_.CallAs<AstroDataCall>();
    if (ast == nullptr)
        return false;

    // if (!this->assertData(inMpdc, outDpdc)) return false;
    ast->SetFrameID(out->FrameID(), true);
    if (!(*ast)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
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
    if (ast == nullptr)
        return false;

    auto* vdc = this->volume_in_slot_.CallAs<geocalls::VolumetricDataCall>();
    if (vdc == nullptr)
        return false;

    auto* tdc = this->temp_in_slot_.CallAs<geocalls::VolumetricDataCall>();
    if (tdc == nullptr)
        return false;

    auto* mdc = this->mass_in_slot_.CallAs<geocalls::VolumetricDataCall>();
    if (mdc == nullptr)
        return false;

    auto* mwdc = this->mw_in_slot_.CallAs<geocalls::VolumetricDataCall>();
    if (mwdc == nullptr)
        return false;

    auto* outVol = dynamic_cast<geocalls::VolumetricDataCall*>(&c);
    if (outVol == nullptr)
        return false;

    ast->SetFrameID(outVol->FrameID(), true);
    vdc->SetFrameID(outVol->FrameID(), true);
    tdc->SetFrameID(outVol->FrameID(), true);
    mdc->SetFrameID(outVol->FrameID(), true);
    mwdc->SetFrameID(outVol->FrameID(), true);
    if (!(*ast)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get extents.");
        return false;
    }
    if (!(*ast)(0)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get data.");
        return false;
    }
    if (!(*vdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get volume metadata.");
        return false;
    }
    if (!(*vdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get volume extents.");
        return false;
    }
    if (!(*vdc)(geocalls::VolumetricDataCall::IDX_GET_DATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get volume data.");
        return false;
    }
    if (!(*tdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get temperature volume metadata.");
        return false;
    }
    if (!(*tdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get temperature volume extents.");
        return false;
    }
    if (!(*tdc)(geocalls::VolumetricDataCall::IDX_GET_DATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get temperature volume data.");
        return false;
    }
    if (!(*mdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get mass volume metadata.");
        return false;
    }
    if (!(*mdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get mass volume extents.");
        return false;
    }
    if (!(*mdc)(geocalls::VolumetricDataCall::IDX_GET_DATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get mass volume data.");
        return false;
    }

    if (!(*mwdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get molecular weight volume metadata.");
        return false;
    }
    if (!(*mwdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get molecular weight volume extents.");
        return false;
    }
    if (!(*mwdc)(geocalls::VolumetricDataCall::IDX_GET_DATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get molecular weight volume data.");
        return false;
    }
    if (this->time != ast->FrameID() || this->time != vdc->FrameID() || this->time != tdc->FrameID() ||
        this->time != mdc->FrameID() || this->time != mwdc->FrameID() || this->in_datahash != ast->DataHash() ||
        this->anythingDirty()) {
        if (!this->createVolumeCPU(*vdc, *tdc, *mdc, *mwdc, *ast))
            return false;
        this->time = ast->FrameID();
        this->in_datahash = ast->DataHash();
        ++this->datahash;
        this->resetDirty();
    }

    // TODO set data
    outVol->SetData(this->vol_[0].data());
    metadata.Components = 1; //< TODO Maybe we want several wavelengths simultaneously
    metadata.GridType = geocalls::GridType_t::CARTESIAN;
    metadata.Resolution[0] = static_cast<size_t>(this->xResSlot.Param<core::param::IntParam>()->Value());
    metadata.Resolution[1] = static_cast<size_t>(this->yResSlot.Param<core::param::IntParam>()->Value());
    metadata.Resolution[2] = static_cast<size_t>(this->zResSlot.Param<core::param::IntParam>()->Value());
    metadata.ScalarType = geocalls::ScalarType_t::FLOATING_POINT;
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
    if (ast == nullptr)
        return false;

    auto* vdc = this->volume_in_slot_.CallAs<geocalls::VolumetricDataCall>();
    if (vdc == nullptr)
        return false;

    auto* tdc = this->temp_in_slot_.CallAs<geocalls::VolumetricDataCall>();
    if (tdc == nullptr)
        return false;

    auto* mdc = this->mass_in_slot_.CallAs<geocalls::VolumetricDataCall>();
    if (mdc == nullptr)
        return false;

    auto* mwdc = this->mw_in_slot_.CallAs<geocalls::VolumetricDataCall>();
    if (mwdc == nullptr)
        return false;

    auto* outVol = dynamic_cast<geocalls::VolumetricDataCall*>(&c);
    if (outVol == nullptr)
        return false;

    ast->SetFrameID(outVol->FrameID(), true);
    vdc->SetFrameID(outVol->FrameID(), true);
    tdc->SetFrameID(outVol->FrameID(), true);
    mdc->SetFrameID(outVol->FrameID(), true);
    mwdc->SetFrameID(outVol->FrameID(), true);
    if (!(*ast)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get extents.");
        return false;
    }
    if (!(*ast)(0)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get data.");
        return false;
    }
    if (!(*vdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get volume metadata.");
        return false;
    }
    if (!(*vdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get volume extents.");
        return false;
    }
    if (!(*vdc)(geocalls::VolumetricDataCall::IDX_GET_DATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get volume data.");
        return false;
    }
    if (!(*tdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get temperature volume metadata.");
        return false;
    }
    if (!(*tdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get temperature volume extents.");
        return false;
    }
    if (!(*tdc)(geocalls::VolumetricDataCall::IDX_GET_DATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get temperature volume data.");
        return false;
    }
    if (!(*mdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get mass volume metadata.");
        return false;
    }
    if (!(*mdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get mass volume extents.");
        return false;
    }
    if (!(*mdc)(geocalls::VolumetricDataCall::IDX_GET_DATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get mass volume data.");
        return false;
    }

    if (!(*mwdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get molecular weight volume metadata.");
        return false;
    }
    if (!(*mwdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get molecular weight volume extents.");
        return false;
    }
    if (!(*mwdc)(geocalls::VolumetricDataCall::IDX_GET_DATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get molecular weight volume data.");
        return false;
    }
    if (this->time != ast->FrameID() || this->time != vdc->FrameID() || this->time != tdc->FrameID() ||
        this->time != mdc->FrameID() || this->time != mwdc->FrameID() || this->in_datahash != ast->DataHash() ||
        this->anythingDirty()) {
        if (!this->createBremsstrahlungVolume(*vdc, *tdc, *mdc, *mwdc, *ast))
            return false;
        this->time = ast->FrameID();
        this->in_datahash = ast->DataHash();
        ++this->datahash;
        this->resetDirty();
    }

    // TODO set data
    outVol->SetData(this->vol_[0].data());
    metadata.Components = 1; //< TODO Maybe we want several wavelengths simultaneously
    metadata.GridType = geocalls::GridType_t::CARTESIAN;
    metadata.Resolution[0] = static_cast<size_t>(this->xResSlot.Param<core::param::IntParam>()->Value());
    metadata.Resolution[1] = static_cast<size_t>(this->yResSlot.Param<core::param::IntParam>()->Value());
    metadata.Resolution[2] = static_cast<size_t>(this->zResSlot.Param<core::param::IntParam>()->Value());
    metadata.ScalarType = geocalls::ScalarType_t::FLOATING_POINT;
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


bool megamol::astro::SpectralIntensityVolume::getAbsorptionDataCallback(core::Call& c) {
    auto* ast = this->astro_in_slot_.CallAs<AstroDataCall>();
    if (ast == nullptr)
        return false;

    auto* vdc = this->volume_in_slot_.CallAs<geocalls::VolumetricDataCall>();
    if (vdc == nullptr)
        return false;

    auto* tdc = this->temp_in_slot_.CallAs<geocalls::VolumetricDataCall>();
    if (tdc == nullptr)
        return false;

    auto* mdc = this->mass_in_slot_.CallAs<geocalls::VolumetricDataCall>();
    if (mdc == nullptr)
        return false;

    auto* mwdc = this->mw_in_slot_.CallAs<geocalls::VolumetricDataCall>();
    if (mwdc == nullptr)
        return false;

    auto* outVol = dynamic_cast<geocalls::VolumetricDataCall*>(&c);
    if (outVol == nullptr)
        return false;

    ast->SetFrameID(outVol->FrameID(), true);
    vdc->SetFrameID(outVol->FrameID(), true);
    tdc->SetFrameID(outVol->FrameID(), true);
    mdc->SetFrameID(outVol->FrameID(), true);
    mwdc->SetFrameID(outVol->FrameID(), true);
    if (!(*ast)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get extents.");
        return false;
    }
    if (!(*ast)(0)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get data.");
        return false;
    }
    if (!(*vdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get volume metadata.");
        return false;
    }
    if (!(*vdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get volume extents.");
        return false;
    }
    if (!(*vdc)(geocalls::VolumetricDataCall::IDX_GET_DATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("SpectralIntensityVolume: Unable to get volume data.");
        return false;
    }
    if (!(*tdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get temperature volume metadata.");
        return false;
    }
    if (!(*tdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get temperature volume extents.");
        return false;
    }
    if (!(*tdc)(geocalls::VolumetricDataCall::IDX_GET_DATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get temperature volume data.");
        return false;
    }
    if (!(*mdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get mass volume metadata.");
        return false;
    }
    if (!(*mdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get mass volume extents.");
        return false;
    }
    if (!(*mdc)(geocalls::VolumetricDataCall::IDX_GET_DATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get mass volume data.");
        return false;
    }

    if (!(*mwdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get molecular weight volume metadata.");
        return false;
    }
    if (!(*mwdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get molecular weight volume extents.");
        return false;
    }
    if (!(*mwdc)(geocalls::VolumetricDataCall::IDX_GET_DATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Unable to get molecular weight volume data.");
        return false;
    }
    if (this->time != ast->FrameID() || this->time != vdc->FrameID() || this->time != tdc->FrameID() ||
        this->time != mdc->FrameID() || this->time != mwdc->FrameID() || this->in_datahash != ast->DataHash() ||
        this->anythingDirty()) {
        if (!this->createAbsorptionVolume(*vdc, *tdc, *mdc, *mwdc, *ast))
            return false;
        this->time = ast->FrameID();
        this->in_datahash = ast->DataHash();
        ++this->datahash;
        this->resetDirty();
    }

    // TODO set data
    outVol->SetData(this->vol_[0].data());
    metadata.Components = 1; //< TODO Maybe we want several wavelengths simultaneously
    metadata.GridType = geocalls::GridType_t::CARTESIAN;
    metadata.Resolution[0] = static_cast<size_t>(this->xResSlot.Param<core::param::IntParam>()->Value());
    metadata.Resolution[1] = static_cast<size_t>(this->yResSlot.Param<core::param::IntParam>()->Value());
    metadata.Resolution[2] = static_cast<size_t>(this->zResSlot.Param<core::param::IntParam>()->Value());
    metadata.ScalarType = geocalls::ScalarType_t::FLOATING_POINT;
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


bool megamol::astro::SpectralIntensityVolume::createVolumeCPU(geocalls::VolumetricDataCall const& volumeIn,
    geocalls::VolumetricDataCall const& tempIn, geocalls::VolumetricDataCall const& massIn,
    geocalls::VolumetricDataCall const& mwIn, AstroDataCall& astroIn) {
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("SpectralIntensityVolume: Starting volume creation.");

    auto const sx = this->xResSlot.Param<core::param::IntParam>()->Value();
    auto const sy = this->yResSlot.Param<core::param::IntParam>()->Value();
    auto const sz = this->zResSlot.Param<core::param::IntParam>()->Value();

    // float const wl = this->wavelength_slot_.Param<core::param::FloatParam>()->Value() / 1000000000.0f;

    auto const numSamples = numSamplesSlot.Param<core::param::IntParam>()->Value();
    double const bias = absorptionBiasSlot.Param<core::param::FloatParam>()->Value();

    auto const numConeSamples = coneSampleNumSlot.Param<core::param::IntParam>()->Value();
    auto const coneAngleDeg = coneAngleSlot.Param<core::param::FloatParam>()->Value();

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
    /*std::transform(dens.cbegin(), dens.cend(), dens.begin(),
        [min_dens, minmax_dens_rcp](auto& val) { return (val - min_dens) * minmax_dens_rcp; });
    std::transform(temps.cbegin(), temps.cend(), temps.begin(),
        [min_temp, minmax_temp_rcp](auto& val) { return (val - min_temp) * minmax_temp_rcp; });*/

    std::vector<double> radiance(dens.size());
    std::transform(dens.cbegin(), dens.cend(), temps.cbegin(), radiance.begin(),
        [](double d, double t) { return d * d * std::sqrt(t); });
    std::transform(sl.cbegin(), sl.cend(), radiance.cbegin(), radiance.begin(),
        [](float r, double rad) { return 4.0 * 0.333333333 * M_PI * rad * static_cast<double>(r * r * r); });
    auto const minmax_rad = std::minmax_element(radiance.cbegin(), radiance.cend());
    auto const min_rad = *minmax_rad.first;
    auto const minmax_rad_rcp = 1.0 / ((*minmax_rad.second) - min_rad);
    std::transform(radiance.cbegin(), radiance.cend(), radiance.begin(),
        [min_rad, minmax_rad_rcp](auto& val) { return (val - min_rad) * minmax_rad_rcp; });


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

    if (vol_sx != temp_vol_sx || vol_sx != mass_vol_sx || vol_sx != mw_vol_sx || vol_sy != temp_vol_sy ||
        vol_sy != mass_vol_sy || vol_sy != mw_vol_sy || vol_sz != temp_vol_sz || vol_sz != mass_vol_sz ||
        vol_sz != mw_vol_sz) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Input volume size not compatible to requested output.");
        return false;
    }

    auto const cell_vol = vol_disx * vol_disy * vol_disz;
    auto const num_cells = vol_sx * vol_sy * vol_sz;

    std::vector<double> optical(num_cells);
    std::transform(volume, volume + numCells, temperature, optical.begin(), [](float d, float t) {
        // offensive formula
        //return 0.018 * std::pow(static_cast<double>(t), -1.5) * 0.0134 * 0.0134 * static_cast<double>(mw) * 1.2;
        // correct formula?
        return 0.018 * std::pow(static_cast<double>(t), -1.5) * 1.4 * static_cast<double>(d) * static_cast<double>(d) *
               1.2;
        // return 1.7 * 10e-25 * std::pow(static_cast<double>(t), -3.5) * 0.0134 * 0.0134 * static_cast<double>(mw)
        // * 1.2;
    });
    std::transform(mass, mass + numCells, optical.cbegin(), optical.begin(), [](float m, double o) {
        // offensive formula
        //return o / m;
        // correct formula
        return o / (static_cast<double>(m) * static_cast<double>(m));
    });
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

    std::vector<glm::i32vec3> voxel_idx(positions.size());
    std::transform(positions.cbegin(), positions.cend(), voxel_idx.begin(),
        [minOSx, minOSy, minOSz, sliceDistX, sliceDistY, sliceDistZ](auto const& pos) {
            return glm::i32vec3(static_cast<int>((pos.x - minOSx) / sliceDistX),
                static_cast<int>((pos.y - minOSy) / sliceDistY), static_cast<int>((pos.z - minOSz) / sliceDistZ));
        });

    /*std::vector<glm::vec3> voxel_pos(numCells);
    for (int vz = 0; vz < sz; ++vz) {
        for (int vy = 0; vy < sy; ++vy) {
            for (int vx = 0; vx < sx; ++vx) {
                auto const idx = (vz * sy + vy) * sx + vx;
                voxel_pos[idx].x = static_cast<float>(vx) * sliceDistX + minOSx;
                voxel_pos[idx].y = static_cast<float>(vy) * sliceDistY + minOSy;
                voxel_pos[idx].z = static_cast<float>(vz) * sliceDistZ + minOSz;
            }
        }
    }*/

    std::uniform_real_distribution<> distr(0.0, 1.0);
    std::mt19937_64 rng(42);

    // Implements the Bump Function from
    // https://en.wikipedia.org/wiki/Radial_basis_function
    auto rbf = [](float const dist, float const epsilon) -> float {
        if (dist >= epsilon)
            return 0.0f;
        return std::exp(-1.0f / (1.0f - std::pow((1.0f / epsilon) * dist, 2.0f)));
    };

#if 1
    cpb.Start("Volume Creation", positions.size());
    auto const cone_factor = std::tan(coneAngleDeg * M_PI / 180.0f);
    auto const cone_angle = coneAngleDeg * M_PI / 180.0;

#pragma omp parallel for
    for (int64_t idx = 0; idx < positions.size(); ++idx) {
        auto const pos = positions[idx];
        /*auto x_base = pos.x;
        auto x = voxel_idx[idx].x;
        auto y_base = pos.y;
        auto y = voxel_idx[idx].y;
        auto z_base = pos.z;
        auto z = voxel_idx[idx].z;*/
        auto const rad = sl[idx];


        for (int iter = 0; iter < numSamples; ++iter) {
            // https://corysimon.github.io/articles/uniformdistn-on-sphere/
            auto phi = 2.0 * M_PI * distr(rng);
            auto theta = std::acos(1.0 - 2.0 * distr(rng));
            glm::vec3 dir = glm::vec3(
                rad * std::sin(theta) * std::cos(phi), rad * std::sin(theta) * std::sin(phi), rad * std::cos(theta));
            glm::vec3 org = pos + dir;
            dir = glm::normalize(dir);
            auto org_dir = dir;

            for (int cone_idx = 0; cone_idx < numConeSamples; ++cone_idx) {
                auto e = radiance[idx];

                // modify dir
                // https://stackoverflow.com/questions/38997302/create-random-unit-vector-inside-a-defined-conical-region
                try {
                    auto const z = distr(rng) * (1.0 - std::cos(cone_angle)) + std::cos(cone_angle);
                    auto const phi = distr(rng) * 2.0 * M_PI;
                    auto const y = std::sqrt(1.0 - z * z) * sin(phi);
                    auto const x = std::sqrt(1.0 - z * z) * cos(phi);
                    glm::vec3 rand(x, y, z);
                    glm::vec3 base(0, 0, 1);
                    // TODO Careful: the next two method calls are adapted from the old new camera and may be broken
                    auto const quat = quat_from_vectors(base, org_dir);
                    dir = quat_rotate(rand, quat);
                } catch (...) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError("SpectralIntensityVolume: Math gone wrong");
                }

                //double att = 0.0;
                float t = 0.0f;
                float t_max = std::sqrt(rangeOSx * rangeOSx + rangeOSy * rangeOSy + rangeOSz * rangeOSz);
                float t_step = min_vol_dis;
                while (t <= t_max && e > 0.0) {
                    glm::vec3 const curr = org + t * dir;

                    auto ax = static_cast<int>((curr.x - vol_orgx) / vol_disx);
                    auto ay = static_cast<int>((curr.y - vol_orgy) / vol_disy);
                    auto az = static_cast<int>((curr.z - vol_orgz) / vol_disz);

                    ax = (ax + 4 * vol_sx) % vol_sx;
                    ay = (ay + 4 * vol_sy) % vol_sy;
                    az = (az + 4 * vol_sz) % vol_sz;

                    double aps = optical[(az * vol_sy + ay) * vol_sx + ax];

                    auto vx = static_cast<int>((curr.x - minOSx) / sliceDistX);
                    auto vy = static_cast<int>((curr.y - minOSy) / sliceDistY);
                    auto vz = static_cast<int>((curr.z - minOSz) / sliceDistZ);

                    vx = (vx + 4 * sx) % sx;
                    vy = (vy + 4 * sy) % sy;
                    vz = (vz + 4 * sz) % sz;

                    e -= e * aps;
                    // att += aps * (1.0 - att);

                    vol_[omp_get_thread_num()][(vz * sy + vy) * sx + vx] += e;

                    /*auto const cone = cone_factor * t;
                    auto const voxel_diff_x = static_cast<int>(cone / sliceDistX);
                    auto const voxel_diff_y = static_cast<int>(cone / sliceDistY);
                    auto const voxel_diff_z = static_cast<int>(cone / sliceDistZ);
                    for (int vvz = vz - voxel_diff_z; vvz < vz + voxel_diff_z; ++vvz) {
                        for (int vvy = vy - voxel_diff_y; vvy < vy + voxel_diff_y; ++vvy) {
                            for (int vvx = vx - voxel_diff_x; vvx < vx + voxel_diff_x; ++vvx) {
                                float const tmp_dis_x = sliceDistX * static_cast<float>(std::abs(vvx - vx));
                                float const tmp_dis_y = sliceDistY * static_cast<float>(std::abs(vvy - vy));
                                float const tmp_dis_z = sliceDistZ * static_cast<float>(std::abs(vvz - vz));
                                auto const distance =
                                    std::sqrt(tmp_dis_x * tmp_dis_x + tmp_dis_y * tmp_dis_y + tmp_dis_z * tmp_dis_z);
                                auto const hvx = (vvx + 2 * sx) % sx;
                                auto const hvy = (vvy + 2 * sy) % sy;
                                auto const hvz = (vvz + 2 * sz) % sz;
                                vol_[omp_get_thread_num()][(hvz * sy + hvy) * sx + hvx] += e * rbf(distance, cone);
                            }
                        }
                    }*/

                    t += t_step;
                }
            }
        }

        ++counter;
        if (omp_get_thread_num() == 0) {
            cpb.Set(counter.load());
        }
    }
    cpb.Stop();
#endif

    for (int i = 1; i < omp_get_max_threads(); ++i) {
        std::transform(vol_[i].begin(), vol_[i].end(), vol_[0].begin(), vol_[0].begin(), std::plus<>());
    }

    max_dens_ = *std::max_element(vol_[0].begin(), vol_[0].end());
    min_dens_ = *std::min_element(vol_[0].begin(), vol_[0].end());
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
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
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("SpectralIntensityVolume: Debug file written\n");
#endif

    // Cleanup
    vol_.resize(1);

    return true;
}


bool megamol::astro::SpectralIntensityVolume::createBremsstrahlungVolume(geocalls::VolumetricDataCall const& volumeIn,
    geocalls::VolumetricDataCall const& tempIn, geocalls::VolumetricDataCall const& massIn,
    geocalls::VolumetricDataCall const& mwIn, AstroDataCall& astroIn) {
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("SpectralIntensityVolume: Starting volume creation.");

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
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Density and temperature volume are not compatible. Aborting.");
        return false;
    }

    if (vol_sx != sx || vol_sy != sy || vol_sz != sz) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
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
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
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


bool megamol::astro::SpectralIntensityVolume::createAbsorptionVolume(geocalls::VolumetricDataCall const& volumeIn,
    geocalls::VolumetricDataCall const& tempIn, geocalls::VolumetricDataCall const& massIn,
    geocalls::VolumetricDataCall const& mwIn, AstroDataCall& astroIn) {
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("SpectralIntensityVolume: Starting volume creation.");

    sx = this->xResSlot.Param<core::param::IntParam>()->Value();
    sy = this->yResSlot.Param<core::param::IntParam>()->Value();
    sz = this->zResSlot.Param<core::param::IntParam>()->Value();

    auto numCells = sx * sy * sz;


    // prepare input mass
    auto metadata = massIn.GetMetadata();
    auto mass = reinterpret_cast<float const*>(massIn.GetData());

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

    // prepare input molecular weight
    metadata = tempIn.GetMetadata();
    auto mw = reinterpret_cast<float const*>(tempIn.GetData());

    auto mw_vol_min = metadata->MinValues[0];
    auto mw_vol_max = metadata->MaxValues[0];

    auto mw_vol_sx = metadata->Resolution[0];
    auto mw_vol_sy = metadata->Resolution[1];
    auto mw_vol_sz = metadata->Resolution[2];


    if (vol_sx != temp_vol_sx || vol_sy != temp_vol_sy || vol_sz != temp_vol_sz || vol_sx != mw_vol_sx ||
        vol_sy != mw_vol_sy || vol_sz != mw_vol_sz) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SpectralIntensityVolume: Density and temperature volume are not compatible. Aborting.");
        return false;
    }

    if (vol_sx != sx || vol_sy != sy || vol_sz != sz) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
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
    std::transform(mw, mw + numCells, temperature, vol_[0].begin(), [](float mw, float t) {
        return 0.018 * std::pow(static_cast<double>(t), -1.5) * 0.0134 * 0.0134 * static_cast<double>(mw) * 1.2;
    });
    std::transform(mass, mass + numCells, vol_[0].cbegin(), vol_[0].begin(), [](float m, double o) { return o / m; });
    auto const minmax_optical = std::minmax_element(vol_[0].cbegin(), vol_[0].cend());
    auto const min_optical = *minmax_optical.first;
    auto const minmax_optical_rcp = 1.0 / (*minmax_optical.second - min_optical);
    std::transform(vol_[0].cbegin(), vol_[0].cend(), vol_[0].begin(),
        [min_optical, minmax_optical_rcp](float o) { return (o - min_optical) * minmax_optical_rcp; });

    max_dens_ = *std::max_element(vol_[0].begin(), vol_[0].end());
    min_dens_ = *std::min_element(vol_[0].begin(), vol_[0].end());
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
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
