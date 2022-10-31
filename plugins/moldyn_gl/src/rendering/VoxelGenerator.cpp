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
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS), &VoxelGenerator::onGenerateVoxels); //TODO change
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

MultiParticleDataCall* VoxelGenerator::getData(unsigned int t, float& out_scaling) {

    MultiParticleDataCall* c = this->get_data_slot_.CallAs<MultiParticleDataCall>();

    // TODO

    return nullptr;
}

