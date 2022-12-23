#include "VoxelGenerator.h"


using namespace megamol::moldyn_gl::rendering;
using namespace megamol::geocalls;

VoxelGenerator::VoxelGenerator(void)
        : core::Module()
        , generate_voxels_slot_("GenerateVoxels", "Slot for requesting voxel generation.")
        , get_data_slot_("GetParticleData", "Connects to the data source") {

    // VolumetricDataCall slot
    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA), &VoxelGenerator::getDataCallback);
    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS), &VoxelGenerator::getExtentCallback);
    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA), &VoxelGenerator::getExtentCallback);
    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_START_ASYNC), &VoxelGenerator::dummyCallback);
    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_STOP_ASYNC), &VoxelGenerator::dummyCallback);
    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_TRY_GET_DATA), &VoxelGenerator::dummyCallback);

    this->MakeSlotAvailable(&this->generate_voxels_slot_);

    // MultiParticleDataCall slot
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

bool VoxelGenerator::getExtentCallback(core::Call& call) {

    // calls
    VolumetricDataCall* volume_call = dynamic_cast<VolumetricDataCall*>(&call);
    MultiParticleDataCall* particle_call = this->get_data_slot_.CallAs<MultiParticleDataCall>();
    if (particle_call == nullptr)
        return false;

    // set frame id
    auto frameID = volume_call != nullptr? volume_call->FrameID() : 0;
    particle_call->SetFrameID(frameID, true);

    // get particle call extents
    if (!(*particle_call)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("VoxelGenerator: could not get current frame extents"); //(% u) ", time-1);
        return false;
    }

    // set volume call data (bounding boxes and frame count)
    if (volume_call != nullptr) {
        volume_call->AccessBoundingBoxes().SetObjectSpaceBBox(particle_call->GetBoundingBoxes().ObjectSpaceBBox());
        volume_call->AccessBoundingBoxes().SetObjectSpaceClipBox(
            particle_call->GetBoundingBoxes().ObjectSpaceClipBox());
        volume_call->AccessBoundingBoxes().MakeScaledWorld(1.0f);
        volume_call->SetFrameCount(particle_call->FrameCount());
    }

    return true;
}

bool VoxelGenerator::getDataCallback(core::Call& call) {

    MultiParticleDataCall* particle_call = this->get_data_slot_.CallAs<MultiParticleDataCall>();
    if (particle_call == nullptr)
        return false;

    VolumetricDataCall* volume_call = dynamic_cast<VolumetricDataCall*>(&call);

    if (volume_call != nullptr) {

        // get frame id
        auto frameID = volume_call != nullptr ? volume_call->FrameID() : 0;

        do {
            particle_call->SetFrameID(frameID, true);
            if (!(*particle_call)(VolumetricDataCall::IDX_GET_EXTENTS)) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("VoxelGenerator: Unable to get extents.");
                return false;
            }
            if (!(*particle_call)(VolumetricDataCall::IDX_GET_DATA)) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("VoxelGenerator: Unable to get data.");
                return false;
            }
        } while (particle_call->FrameID() != frameID);

        // TODO time, datahash, dirty (ParticleToDensity.cpp)
        if (!this->generateVoxels(particle_call))
            return false;
        auto time = particle_call->FrameID();

        // TODO create volume and metadata somewhere else!!
        std::vector<std::vector<float>> vol; // TODO


        // set metadata
        metadata.Components = 3; //? // is_vector ? 3 : 1;
        metadata.GridType = GridType_t::CARTESIAN;
        //metadata.Resolution[0] = xRes; // TODO set parameters in UI?
        //metadata.Resolution[1] = yRes;
        //metadata.Resolution[2] = zRes;
        metadata.ScalarType = ScalarType_t::FLOATING_POINT;
        metadata.ScalarLength = sizeof(float);
        //TODO metadata.MinValues and metadata.MaxValues
        auto bbox = particle_call->AccessBoundingBoxes().ObjectSpaceBBox();
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

        metadata.MemLoc = MemoryLocation::VRAM;

        // set volume call data
        volume_call->SetFrameID(time);
        //volume_call->SetData(vol[0].data());
        volume_call->SetMetadata(&metadata);
        //volume_call->SetDataHash(this->datahash); // TODO

    }

    return true;
}


bool VoxelGenerator::generateVoxels(MultiParticleDataCall* particle_call) {

    // TODO

    // see SphereRenderer.cpp


    //----------------------------------------
    // use MDAOVolumeGenerator? (vol_gen_)
 
    //  SetShaderSourceFactory(..)
    //  Init(..)
 
    //  SetResolution(width, height, depth)
    //  ClearVolume()
    //  StartInsertion(..)
    //  for
    //      InsertParticles(..)
    //  EndInsertion()
    //  RecreateMipMap()

    // GetVolumeTextureHandle()

    //----------------------------------------



    // inDensityTex



    return true;
}

bool VoxelGenerator::dummyCallback(core::Call& call) {
    return true;
}
