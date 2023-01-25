#include "VoxelGenerator.h"

#include "OpenGL_Context.h"
#include "mmcore/param/IntParam.h"

using namespace megamol::moldyn_gl::rendering;
using namespace megamol::geocalls;

VoxelGenerator::VoxelGenerator(void)
        : mmstd_gl::ModuleGL()
        , generate_voxels_slot_("GenerateVoxels", "Slot for requesting voxel generation.")
        , get_data_slot_("GetParticleData", "Connects to the data source")
        , vol_size_slot_("volumeSize", "Longest volume edge")
        , texture_handle()
        , vol_gen_(nullptr)
        , vertex_array_() {

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

    this->vol_size_slot_ << (new core::param::IntParam(256, 1, 1024));
    this->MakeSlotAvailable(&this->vol_size_slot_);
}

VoxelGenerator::~VoxelGenerator(void) {
    this->Release();
}

bool VoxelGenerator::create(void) {
    this->vol_size_slot_.Param<core::param::IntParam>()->SetGUIVisible(true);
    glGenVertexArrays(1, &vertex_array_);

    return initVolumeGenerator();

    //return true;
}

void VoxelGenerator::release(void) {

    glDeleteVertexArrays(1, &vertex_array_);

    //TODO reset resources
    this->vol_size_slot_.Param<core::param::IntParam>()->SetGUIVisible(false);

    if (this->vol_gen_ != nullptr) {
        delete this->vol_gen_;
        this->vol_gen_ = nullptr;
    }
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

bool VoxelGenerator::initVolumeGenerator() {

    auto shader_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
    vol_gen_ = new misc::MDAOVolumeGenerator();
    vol_gen_->SetShaderSourceFactory(&shader_options);

    // Init volume generator
    if (!vol_gen_->Init(frontend_resources.get<frontend_resources::OpenGL_Context>())) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error initializing volume generator. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}

bool VoxelGenerator::generateVoxels(MultiParticleDataCall* particle_call) {

    // make sure volume generator is initialized
    if (this->vol_gen_ == nullptr) {
        initVolumeGenerator();
    }
    
    glm::vec4 cur_clip_dat_ = glm::vec4(0.0);                                   //TODO
    vislib::math::Cuboid<float> cur_clip_box_(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0); // TODO sphererenderer: AccessBoundingBoxes().ClipBox(); (CallRender3DGL)

    // Fill volume texture
    if (vol_gen_ != nullptr) {
        vislib::math::Dimension<float, 3> dims = cur_clip_box_.GetSize();
        int vol_size = this->vol_size_slot_.Param<core::param::IntParam>()->Value();

        float longest_edge = cur_clip_box_.LongestEdge();
        dims.Scale(static_cast<float>(vol_size)/longest_edge);
        dims.SetWidth(ceil(dims.GetWidth() / 4.0f) * 4.0f);
        dims.SetHeight(ceil(dims.GetHeight()));
        dims.SetDepth(ceil(dims.GetDepth()));


        vol_gen_->SetResolution(dims.GetWidth(), dims.GetHeight(), dims.GetDepth());
        vol_gen_->ClearVolume();
        vol_gen_->StartInsertion(
            cur_clip_box_, glm::vec4(cur_clip_dat_[0], cur_clip_dat_[1], cur_clip_dat_[2], cur_clip_dat_[3]));

        // Insert particle data
        unsigned int particleListCount = particle_call->GetParticleListCount(); 
        for (unsigned int i = 0; i < particleListCount; i++) {
            float global_radius = 0.0f;

            auto particles = particle_call->AccessParticles(i);

            if (particles.GetVertexDataType() !=
                MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR)
                global_radius = particles.GetGlobalRadius();


            //// TODO one vertex array for each i?
            //glBindVertexArray(vertex_array_);
            ////enableBufferData(prgm, parts, this->gpu_data_[i].vertex_vbo, parts.GetVertexData(),this->gpu_data_[i].color_vbo, parts.GetColourData(), true)
            //glBindVertexArray(0);
            ////disableBufferData

            //TODO vertex array!!

            vol_gen_->InsertParticles(
                static_cast<unsigned int>(particles.GetCount()), global_radius, 3); // TODO, 3: handle for vertex array, SphereRenderer: this->gpu_data_[i].vertex_array
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

//TODO use?
void VoxelGenerator::getClipData(glm::vec4& out_clip_dat, glm::vec4& out_clip_col) {
    // TODO if call get_clip_plane_slot_
    // else:
    out_clip_dat[0] = out_clip_dat[1] = out_clip_dat[2] = out_clip_dat[3] = 0.0f;
    out_clip_col[0] = out_clip_col[1] = out_clip_col[2] = 0.75f;
    out_clip_col[3] = 1.0f;
}
