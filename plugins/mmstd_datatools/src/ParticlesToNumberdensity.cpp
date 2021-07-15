#include "ParticlesToNumberdensity.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"

#include "glm/glm.hpp"


megamol::stdplugin::datatools::ParticlesToNumberdensity::ParticlesToNumberdensity()
        : data_out_slot_("dataOut", "")
        , data_in_slot_("dataIn", "")
        , grid_x_res_slot_("gridX", "")
        , grid_y_res_slot_("gridY", "")
        , grid_z_res_slot_("gridZ", "")
        , surface_slot_("forSurfaceReconstruction", "Set true if this volume is used for surface reconstruction") {
    data_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_DATA),
        &ParticlesToNumberdensity::get_data_cb);
    data_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_EXTENTS),
        &ParticlesToNumberdensity::get_extent_cb);
    data_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_METADATA),
        &ParticlesToNumberdensity::get_extent_cb);
    data_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_START_ASYNC),
        &ParticlesToNumberdensity::dummy_cb);
    data_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_STOP_ASYNC),
        &ParticlesToNumberdensity::dummy_cb);
    data_out_slot_.SetCallback(core::misc::VolumetricDataCall::ClassName(),
        core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_TRY_GET_DATA),
        &ParticlesToNumberdensity::dummy_cb);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    grid_x_res_slot_ << new core::param::IntParam(16, 1);
    MakeSlotAvailable(&grid_x_res_slot_);

    grid_y_res_slot_ << new core::param::IntParam(16, 1);
    MakeSlotAvailable(&grid_y_res_slot_);

    grid_z_res_slot_ << new core::param::IntParam(16, 1);
    MakeSlotAvailable(&grid_z_res_slot_);

    surface_slot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&surface_slot_);
}


megamol::stdplugin::datatools::ParticlesToNumberdensity::~ParticlesToNumberdensity() {
    this->Release();
}


bool megamol::stdplugin::datatools::ParticlesToNumberdensity::create() {
    return true;
}


void megamol::stdplugin::datatools::ParticlesToNumberdensity::release() {}


bool megamol::stdplugin::datatools::ParticlesToNumberdensity::get_data_cb(core::Call& c) {
    auto* inMpdc = this->data_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (inMpdc == nullptr)
        return false;

    auto* outVol = dynamic_cast<core::misc::VolumetricDataCall*>(&c);

    if (outVol != nullptr) {
        auto frameID = outVol->FrameID();
        do {
            inMpdc->SetFrameID(frameID, true);
            if (!(*inMpdc)(1)) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("ParticlesToDensity: Unable to get extents.");
                return false;
            }
            if (!(*inMpdc)(0)) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("ParticlesToDensity: Unable to get data.");
                return false;
            }
        } while (inMpdc->FrameID() != frameID);
        if (this->frame_id_ != inMpdc->FrameID() || this->in_data_hash_ != inMpdc->DataHash() || this->is_dirty()) {
            if (surface_slot_.Param<core::param::BoolParam>()->Value())
                modify_bbox(*inMpdc);
            if (!this->assert_data(*inMpdc))
                return false;
            this->frame_id_ = inMpdc->FrameID();
            this->in_data_hash_ = inMpdc->DataHash();
            ++this->out_data_hash_;
            this->reset_dirty();
        }
    }

    // TODO set data
    if (outVol != nullptr) {
        outVol->SetFrameID(this->frame_id_);
        outVol->SetData(this->vol_data_.data());
        metadata.Components = 1;
        metadata.GridType = core::misc::GridType_t::CARTESIAN;
        metadata.Resolution[0] = static_cast<size_t>(this->grid_x_res_slot_.Param<core::param::IntParam>()->Value());
        metadata.Resolution[1] = static_cast<size_t>(this->grid_y_res_slot_.Param<core::param::IntParam>()->Value());
        metadata.Resolution[2] = static_cast<size_t>(this->grid_z_res_slot_.Param<core::param::IntParam>()->Value());
        metadata.ScalarType = core::misc::ScalarType_t::FLOATING_POINT;
        metadata.ScalarLength = sizeof(float);
        metadata.MinValues = new double[1];
        metadata.MinValues[0] = this->min_dens_;
        metadata.MaxValues = new double[1];
        metadata.MaxValues[0] = this->max_dens_;
        auto bbox = inMpdc->AccessBoundingBoxes().ObjectSpaceBBox();
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
        //-metadata.SliceDists[0][0] / 4.0f;
        metadata.Origin[1] = bbox.Bottom();
        //-metadata.SliceDists[1][0] / 4.0f;
        metadata.Origin[2] = bbox.Back();
        //-metadata.SliceDists[2][0] / 4.0f;

        metadata.IsUniform[0] = true;
        metadata.IsUniform[1] = true;
        metadata.IsUniform[2] = true;
        outVol->SetMetadata(&metadata);

        outVol->SetDataHash(this->out_data_hash_);

        /*outVol->SetVolumeDimension(this->xResSlot.Param<core::param::IntParam>()->Value(),
            this->yResSlot.Param<core::param::IntParam>()->Value(),
        this->zResSlot.Param<core::param::IntParam>()->Value()); outVol->SetComponents(1);
        outVol->SetMinimumDensity(0.0f);
        outVol->SetMaximumDensity(this->maxDens);
        outVol->SetVoxelMapPointer(this->vol[0].data());*/
        // inMpdc->Unlock();
    }

    return true;
}


bool megamol::stdplugin::datatools::ParticlesToNumberdensity::get_extent_cb(core::Call& c) {
    using megamol::core::moldyn::MultiParticleDataCall;

    auto* out = dynamic_cast<core::misc::VolumetricDataCall*>(&c);

    auto* inMpdc = this->data_in_slot_.CallAs<MultiParticleDataCall>();
    if (inMpdc == nullptr)
        return false;

    auto frameID = out->FrameID();
    // vislib::sys::Log::DefaultLog.WriteInfo(L"ParticleToDensity requests frame %u.", frameID);
    inMpdc->SetFrameID(frameID, true);
    if (!(*inMpdc)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "ParticlesToDensity: could not get current frame extents (%u)", frame_id_ - 1);
        return false;
    }

    if (out != nullptr) {
        out->AccessBoundingBoxes().SetObjectSpaceBBox(inMpdc->GetBoundingBoxes().ObjectSpaceBBox());
        out->AccessBoundingBoxes().SetObjectSpaceClipBox(inMpdc->GetBoundingBoxes().ObjectSpaceClipBox());
        out->AccessBoundingBoxes().MakeScaledWorld(1.0f);
        out->SetFrameCount(inMpdc->FrameCount());
    }

    // TODO: what am I actually doing here
    // inMpdc->SetUnlocker(nullptr, false);
    // inMpdc->Unlock();

    return true;
}


bool megamol::stdplugin::datatools::ParticlesToNumberdensity::dummy_cb(core::Call& c) {
    return true;
}


bool megamol::stdplugin::datatools::ParticlesToNumberdensity::assert_data(core::moldyn::MultiParticleDataCall& parts) {
    auto const grid_x_res = grid_x_res_slot_.Param<core::param::IntParam>()->Value();
    auto const grid_y_res = grid_y_res_slot_.Param<core::param::IntParam>()->Value();
    auto const grid_z_res = grid_z_res_slot_.Param<core::param::IntParam>()->Value();

    auto const num_cells = grid_x_res * grid_y_res * grid_z_res;

    vol_data_.resize(num_cells);
    std::fill(vol_data_.begin(), vol_data_.end(), 0.f);

    auto const bbox = parts.AccessBoundingBoxes().ObjectSpaceBBox();
    auto const origin = glm::vec3(bbox.GetLeft(), bbox.GetBottom(), bbox.GetBack());
    auto const dimension = glm::vec3(bbox.Width(), bbox.Height(), bbox.Depth());

    auto const resolution = glm::vec3(grid_x_res, grid_y_res, grid_z_res) - glm::vec3(1.f);

    auto const cell_size = dimension / glm::vec3(grid_x_res, grid_y_res, grid_z_res);
    auto const cell_volume = cell_size.x * cell_size.y * cell_size.z;

    auto const pl_count = parts.GetParticleListCount();

    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& particles = parts.AccessParticles(pl_idx);

        auto const p_count = particles.GetCount();

        auto const x_acc = particles.GetParticleStore().GetXAcc();
        auto const y_acc = particles.GetParticleStore().GetYAcc();
        auto const z_acc = particles.GetParticleStore().GetZAcc();

        for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
            auto const pos = glm::vec3(x_acc->Get_f(p_idx), y_acc->Get_f(p_idx), z_acc->Get_f(p_idx));

            auto const coords = ((pos - origin) / dimension) * resolution;

            auto const u_coords = glm::uvec3(coords.x, coords.y, coords.z);

            vol_data_[u_coords.x + grid_x_res * (u_coords.y + u_coords.z * grid_y_res)] += 1.0f;
        }
    }

    std::for_each(vol_data_.begin(), vol_data_.end(), [&cell_volume](auto& el) { el /= cell_volume; });

    auto const minmax = std::minmax_element(vol_data_.begin(), vol_data_.end());

    min_dens_ = *minmax.first;
    max_dens_ = *minmax.second;

    core::utility::log::Log::DefaultLog.WriteInfo("[ParticlesToNumberdensity] min %f / max %f", min_dens_, max_dens_);

    return true;
}

void megamol::stdplugin::datatools::ParticlesToNumberdensity::modify_bbox(
    megamol::core::moldyn::MultiParticleDataCall& parts) {
    auto sx = this->grid_x_res_slot_.Param<core::param::IntParam>()->Value();
    auto sy = this->grid_y_res_slot_.Param<core::param::IntParam>()->Value();
    auto sz = this->grid_z_res_slot_.Param<core::param::IntParam>()->Value();

    auto rangeOSx = parts.AccessBoundingBoxes().ObjectSpaceBBox().Width();
    auto rangeOSy = parts.AccessBoundingBoxes().ObjectSpaceBBox().Height();
    auto rangeOSz = parts.AccessBoundingBoxes().ObjectSpaceBBox().Depth();

    float general_box_scaling = 1.1;

    // extend deph
    auto spacing = (rangeOSz * general_box_scaling) / sz;
    auto newDepth = (rangeOSz * general_box_scaling) + 2 * spacing;
    spacing = newDepth / sz;

    // ensure cubic voxels
    auto newWidth = (rangeOSx * general_box_scaling) + 2 * spacing;
    int resolutionX = newWidth / spacing;
    auto rest = newWidth / spacing - static_cast<float>(resolutionX);
    newWidth += (1 - rest) * spacing;
    resolutionX += 1;
    this->grid_x_res_slot_.Param<core::param::IntParam>()->SetValue(resolutionX);

    auto newHeight = (rangeOSy * general_box_scaling) + 2 * spacing;
    int resolutionY = newHeight / spacing;
    rest = newHeight / spacing - static_cast<float>(resolutionY);
    newHeight += (1 - rest) * spacing;
    resolutionY += 1;
    this->grid_y_res_slot_.Param<core::param::IntParam>()->SetValue(resolutionY);

    auto minOSx = parts.AccessBoundingBoxes().ObjectSpaceBBox().Left();
    auto minOSy = parts.AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
    auto minOSz = parts.AccessBoundingBoxes().ObjectSpaceBBox().Back();

    auto maxOSx = parts.AccessBoundingBoxes().ObjectSpaceBBox().Right();
    auto maxOSy = parts.AccessBoundingBoxes().ObjectSpaceBBox().Top();
    auto maxOSz = parts.AccessBoundingBoxes().ObjectSpaceBBox().Front();

    auto newLeft = minOSx - (newWidth - rangeOSx) / 2;
    auto newBottom = minOSy - (newHeight - rangeOSy) / 2;
    auto newBack = minOSz - (newDepth - rangeOSz) / 2;

    auto newRight = maxOSx + (newWidth - rangeOSx) / 2;
    auto newTop = maxOSy + (newHeight - rangeOSy) / 2;
    auto newFront = maxOSz + (newDepth - rangeOSz) / 2;

    parts.AccessBoundingBoxes().SetObjectSpaceBBox(newLeft, newBottom, newBack, newRight, newTop, newFront);
    parts.AccessBoundingBoxes().SetObjectSpaceClipBox(newLeft, newBottom, newBack, newRight, newTop, newFront);
}
