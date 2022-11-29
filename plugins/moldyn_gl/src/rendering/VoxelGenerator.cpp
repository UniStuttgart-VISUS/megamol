#include "VoxelGenerator.h"


using namespace megamol::moldyn_gl::rendering;
using namespace megamol::geocalls;

VoxelGenerator::VoxelGenerator(void)
        : core::Module()
        , generate_voxels_slot_("GenerateVoxels", "Slot for requesting voxel generation.")
        , get_data_slot_("GetParticleData", "Connects to the data source") {


    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA), &VoxelGenerator::onGenerateVoxels);
    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS), &VoxelGenerator::GetExtents); //TODO change
    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA), &VoxelGenerator::onGenerateVoxels); //TODO change
    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_START_ASYNC), &VoxelGenerator::onGenerateVoxels); //TODO change
    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_STOP_ASYNC), &VoxelGenerator::onGenerateVoxels); //TODO change
    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_TRY_GET_DATA), &VoxelGenerator::onGenerateVoxels); //TODO change

    this->MakeSlotAvailable(&this->generate_voxels_slot_);

    this->get_data_slot_.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->get_data_slot_.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->get_data_slot_);
}

VoxelGenerator::~VoxelGenerator(void) {
    this->Release();
}

bool VoxelGenerator::create(void) {
    return true; //TODO
}

void VoxelGenerator::release(void) {
    //TODO reset resources
}

bool VoxelGenerator::onGenerateVoxels(core::Call& call) {
    return false;
}

bool VoxelGenerator::GetExtents(core::Call& call) {
    auto cr = &call;
    if (cr == nullptr)
        return false;

    //try { //VolumetricDataSource.cpp
    //    VolumetricDataCall& c = dynamic_cast<VolumetricDataCall&>(call);

    //    c.SetExtent((unsigned int)this->metadata.NumberOfFrames, this->metadata.Origin[0], this->metadata.Origin[1],
    //        this->metadata.Origin[2], this->metadata.Extents[0] + this->metadata.Origin[0],
    //        this->metadata.Extents[1] + this->metadata.Origin[1], this->metadata.Extents[2] + this->metadata.Origin[2]);
    //}
    //catch (...) {
    //    return false;
    //}

    MultiParticleDataCall* c2 = this->get_data_slot_.CallAs<MultiParticleDataCall>();
    /*if ((c2 != nullptr)) {
        c2->SetFrameID(
            static_cast<unsigned int>(cr->Time()), this->force_time_slot_.Param<param::BoolParam>()->Value());
        if (!(*c2)(1))
            return false;
        cr->SetTimeFramesCount(c2->FrameCount());
        auto const plcount = c2->GetParticleListCount();
        if (this->use_local_bbox_param_.Param<param::BoolParam>()->Value() && plcount > 0) {
            auto bbox = c2->AccessParticles(0).GetBBox();
            auto cbbox = bbox;
            cbbox.Grow(c2->AccessParticles(0).GetGlobalRadius());
            for (unsigned pidx = 1; pidx < plcount; ++pidx) {
                auto temp = c2->AccessParticles(pidx).GetBBox();
                bbox.Union(temp);
                temp.Grow(c2->AccessParticles(pidx).GetGlobalRadius());
                cbbox.Union(temp);
            }
            cr->AccessBoundingBoxes().SetBoundingBox(bbox);
            cr->AccessBoundingBoxes().SetClipBox(cbbox);
        } else {
            cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();
        }

    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }
    this->cur_clip_box_ = cr->AccessBoundingBoxes().ClipBox();*/

    return true;
}

MultiParticleDataCall* VoxelGenerator::getData(unsigned int t, float& out_scaling) {

    MultiParticleDataCall* c = this->get_data_slot_.CallAs<MultiParticleDataCall>();

    // TODO

    return nullptr;
}

