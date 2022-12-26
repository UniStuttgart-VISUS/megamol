#include "VoxelGenerator.h"

#include "misc/MDAOVolumeGenerator.h"

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

        volume_call->SetData(texture_handle);
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


    // from SphereRenderer.cpp

    
    
    // TODO get real shader options...
    std::filesystem::path p1 = "D:\\Hiwi\\VISUS\\1_megamol\\sergejs_fork\\megamol\\out\\install\\x64-Debug\\bin\\";
    std::filesystem::path p2 = "D:\\Hiwi\\VISUS\\1_megamol\\sergejs_fork\\megamol\\out\\install\\x64-Debug\\bin\\../share/shaders";
    std::vector<std::filesystem::path> include_paths = {p1, p2};
    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(include_paths); // TODO

    // TODO init only once?
    // Init volume generator
    auto vol_gen_ = new misc::MDAOVolumeGenerator();
    auto so = shader_options;
    vol_gen_->SetShaderSourceFactory(&so);
    //auto context = frontend_resources.get<frontend_resources::OpenGL_Context>(); // error
    auto context = frontend_resources::OpenGL_Context(); // TODO
    if (!vol_gen_->Init(context)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error initializing volume generator. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }


    
    // Check if voxelization is even needed

    // Recreate the volume if neccessary
    bool equal_clip_data = true;
    //for (size_t i = 0; i < 4; i++) {
    //    if (this->old_clip_dat_[i] != this->cur_clip_dat_[i]) {
    //        equal_clip_data = false;
    //        break;
    //    }
    //}
 
    //if ((vol_gen_ != nullptr) && (this->state_invalid_ || this->ao_vol_size_slot_.IsDirty() || !equal_clip_data)) {
    //    this->ao_vol_size_slot_.ResetDirty();

    //    int vol_size = this->ao_vol_size_slot_.Param<param::IntParam>()->Value();

    //    vislib::math::Dimension<float, 3> dims = this->cur_clip_box_.GetSize();

    //    // Calculate the extensions of the volume by using the specified number of voxels for the longest edge
    //    float longest_edge = this->cur_clip_box_.LongestEdge();
    //    dims.Scale(static_cast<float>(vol_size) / longest_edge);

    //    // The X size must be a multiple of 4, so we might have to correct that a little
    //    dims.SetWidth(ceil(dims.GetWidth() / 4.0f) * 4.0f);
    //    dims.SetHeight(ceil(dims.GetHeight()));
    //    dims.SetDepth(ceil(dims.GetDepth()));
    //    this->amb_cone_constants_[0] = std::min(dims.Width(), std::min(dims.Height(), dims.Depth()));
    //    this->amb_cone_constants_[1] = ceil(std::log2(static_cast<float>(vol_size))) - 1.0f;

    //    // Set resolution accordingly
    //    vol_gen_->SetResolution(dims.GetWidth(), dims.GetHeight(), dims.GetDepth());

    //    // Insert all particle lists
    //    vol_gen_->ClearVolume();
    //    vol_gen_->StartInsertion(this->cur_clip_box_,
    //        glm::vec4(this->cur_clip_dat_[0], this->cur_clip_dat_[1], this->cur_clip_dat_[2], this->cur_clip_dat_[3]));

    //    for (unsigned int i = 0; i < this->gpu_data_.size(); i++) {
    //        float global_radius = 0.0f;
    //        if (mpdc->AccessParticles(i).GetVertexDataType() != MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR)
    //            global_radius = mpdc->AccessParticles(i).GetGlobalRadius();

    //        vol_gen_->InsertParticles(static_cast<unsigned int>(mpdc->AccessParticles(i).GetCount()),
    //            global_radius, this->gpu_data_[i].vertex_array);
    //    }
    //    vol_gen_->EndInsertion();

    //    vol_gen_->RecreateMipmap();
    //}


    // tests:
    glm::vec4 cur_clip_dat_; //TODO
    vislib::math::Cuboid<float> cur_clip_box_(-1.0,-1.0,-1.0, 1.0, 1.0, 1.0); // TODO

    if (vol_gen_ != nullptr) {
        vislib::math::Dimension<float, 3> dims = cur_clip_box_.GetSize();
        vol_gen_->SetResolution(dims.GetWidth(), dims.GetHeight(), dims.GetDepth()); // clipbox dimensions
        vol_gen_->ClearVolume();
        vol_gen_->StartInsertion(
            cur_clip_box_, glm::vec4(cur_clip_dat_[0], cur_clip_dat_[1], cur_clip_dat_[2], cur_clip_dat_[3]));


        for (unsigned int i = 0; i < 1; i++) { // TODO
            float global_radius = 0.0f;
            if (particle_call->AccessParticles(i).GetVertexDataType() !=
                MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR)
                global_radius = particle_call->AccessParticles(i).GetGlobalRadius();
            vol_gen_->InsertParticles(
                static_cast<unsigned int>(particle_call->AccessParticles(i).GetCount()), global_radius, 3); // TODO
        }
        
        vol_gen_->EndInsertion();
        vol_gen_->RecreateMipmap();
    }

    // texture handle
    texture_handle = vol_gen_->GetVolumeTextureHandle();

    return true;
}

bool VoxelGenerator::dummyCallback(core::Call& call) {
    return true;
}
