#include "VoxelGenerator.h"


using namespace megamol::moldyn_gl::rendering;

VoxelGenerator::VoxelGenerator(void)
        : core::Module()
        , slotGetVoxels("GetVoxels", "Slot for requesting voxel generation.") {
    //this->slotGetVoxels.SetCallback(VolumetricDataCall::ClassName(), );
    this->MakeSlotAvailable(&this->slotGetVoxels);
}

VoxelGenerator::~VoxelGenerator(void) {
    this->Release();
}

bool megamol::moldyn_gl::rendering::VoxelGenerator::create(void) {
    return true; //TODO
}

void megamol::moldyn_gl::rendering::VoxelGenerator::release(void) {
    //TODO reset resources
}

//MultiParticleDataCall* VoxelGenerator::getData(unsigned int t, float& out_scaling) {
//    // TODO
//    return nullptr;
//}
