#include "VoxelGenerator.h"

#include "OpenGL_Context.h"
#include "mmcore/param/IntParam.h"

using namespace megamol::moldyn_gl::rendering;
using namespace megamol::geocalls;

VoxelGenerator::VoxelGenerator(void)
        : mmstd_gl::ModuleGL()
        , generate_voxels_slot_("generateVoxels", "Slot for requesting voxel generation.")
        , get_data_slot_("getParticleData", "Connects to the data source")
        , vol_size_slot_("volumeSize", "Longest volume edge")
        , texture_handle()
        , vol_gen_(nullptr)
        , vertex_array_()
        , shader_options_flags_(nullptr)
        , sphere_prgm_()
        , vbo_() {

    // VolumetricDataCall slot
    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA), &VoxelGenerator::getDataCallback);
    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS), &VoxelGenerator::getExtentCallback);
    this->generate_voxels_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA), &VoxelGenerator::getMetadataCallback);
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

    this->release();
}

void VoxelGenerator::release(void) {
    this->vol_size_slot_.Param<core::param::IntParam>()->SetGUIVisible(false);
    glDeleteVertexArrays(1, &vertex_array_);
    glDeleteBuffers(1, &vbo_);

    if (this->vol_gen_ != nullptr) {
        delete this->vol_gen_;
        this->vol_gen_ = nullptr;
    }
}

bool VoxelGenerator::create(void) {

    this->vol_size_slot_.Param<core::param::IntParam>()->SetGUIVisible(true);
    glGenVertexArrays(1, &vertex_array_);
    glGenBuffers(1, &vbo_);

    return initVolumeGenerator();
}

bool VoxelGenerator::getExtentCallback(core::Call& call) {

    // calls
    VolumetricDataCall* volume_call = dynamic_cast<VolumetricDataCall*>(&call);
    MultiParticleDataCall* particle_call = this->get_data_slot_.CallAs<MultiParticleDataCall>();
    if (particle_call == nullptr)
        return false;

    // set frame id
    auto frameID = volume_call != nullptr ? volume_call->FrameID() : 0;
    particle_call->SetFrameID(frameID, true);

    // get particle call extents
    if (!(*particle_call)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "VoxelGenerator: could not get current frame extents"); //(% u) ", time-1);
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


bool VoxelGenerator::getMetadataCallback(core::Call& call) {

    VolumetricDataCall* volume_call = dynamic_cast<VolumetricDataCall*>(&call);
    MultiParticleDataCall* particle_call = this->get_data_slot_.CallAs<MultiParticleDataCall>();
    if (particle_call == nullptr || volume_call == nullptr)
        return false;

    // set metadata
    int vol_size = this->vol_size_slot_.Param<core::param::IntParam>()->Value();
    metadata.Resolution[0] = vol_size;
    metadata.Resolution[1] = vol_size;
    metadata.Resolution[2] = vol_size;
    metadata.MemLoc = MemoryLocation::VRAM;
    if (!(*particle_call)(0)) {
        return false;
    }
    auto cbox = particle_call->AccessBoundingBoxes().ClipBox();
    auto orig = cbox.GetOrigin();
    auto size = cbox.GetSize();
    for (int d = 0; d < 3; ++d) {
        metadata.Origin[d] = orig[d];
        metadata.Extents[d] = size[d];
    }

    volume_call->SetMetadata(&metadata);

    return true;
}


bool VoxelGenerator::getDataCallback(core::Call& call) {

    MultiParticleDataCall* particle_call = this->get_data_slot_.CallAs<MultiParticleDataCall>();
    //geocalls::EllipsoidalParticleDataCall* particle_call2 = this->get_data_slot_.CallAs<geocalls::EllipsoidalParticleDataCall>(); // for glyph renderer

    if (particle_call == nullptr)
        return false;

    VolumetricDataCall* volume_call = dynamic_cast<VolumetricDataCall*>(&call);

    if (volume_call != nullptr) {

        // get frame id
        auto frameID = volume_call->FrameID();

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
        } while (particle_call->FrameID() != frameID); // TODO is do while necessary?

        // TODO time, datahash, dirty (ParticleToDensity.cpp)
        if (!this->generateVoxels(particle_call, volume_call))
            return false;

        // set volume call data
        volume_call->SetFrameID(particle_call->FrameID());
        volume_call->SetData(texture_handle);
    }

    return true;
}

bool VoxelGenerator::initVolumeGenerator() {

    auto shader_options = core::utility::make_path_shader_options(
        frontend_resources.get<megamol::frontend_resources::
                RuntimeConfig>()); //msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
    vol_gen_ = new misc::MDAOVolumeGenerator();
    vol_gen_->SetShaderSourceFactory(&shader_options);

    shader_options_flags_ = std::make_unique<msf::ShaderFactoryOptionsOpenGL>(shader_options);
    sphere_prgm_ = core::utility::make_glowl_shader("sphere_mdao", *shader_options_flags_,
        "moldyn_gl/sphere_renderer/sphere_mdao.vert.glsl", "moldyn_gl/sphere_renderer/sphere_mdao.frag.glsl");

    // Init volume generator
    if (!vol_gen_->Init(frontend_resources.get<frontend_resources::OpenGL_Context>())) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error initializing volume generator. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}

bool VoxelGenerator::generateVoxels(MultiParticleDataCall* particle_call, VolumetricDataCall* volume_call) {

    // make sure volume generator is initialized
    if (this->vol_gen_ == nullptr) {
        initVolumeGenerator();
    }

    vislib::math::Cuboid<float> cur_clip_box_ = volume_call->AccessBoundingBoxes().ClipBox();

    // Fill volume texture
    if (vol_gen_ != nullptr) {
        vislib::math::Dimension<float, 3> dims = cur_clip_box_.GetSize();
        int vol_size = this->vol_size_slot_.Param<core::param::IntParam>()->Value();

        float longest_edge = cur_clip_box_.LongestEdge();
        dims.Scale(static_cast<float>(vol_size) / longest_edge);
        dims.SetWidth(ceil(dims.GetWidth() / 4.0f) * 4.0f);
        dims.SetHeight(ceil(dims.GetHeight()));
        dims.SetDepth(ceil(dims.GetDepth()));


        vol_gen_->SetResolution(dims.GetWidth(), dims.GetHeight(), dims.GetDepth());
        vol_gen_->ClearVolume();
        vol_gen_->StartInsertion(cur_clip_box_, glm::vec4(0.0));

        // Insert particle data
        unsigned int particleListCount = particle_call->GetParticleListCount();
        for (unsigned int i = 0; i < particleListCount; i++) {
            float global_radius = 0.0f;

            auto particles = particle_call->AccessParticles(i);

            if (particles.GetVertexDataType() != MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR)
                global_radius = particles.GetGlobalRadius();


            // vao with particle data
            glBindVertexArray(vertex_array_); // TODO different vaos and vbos for different particle lists?
            fillVAO(particles, vbo_, particles.GetVertexData(), true); // shpererenderer enableBuffersData()
            glBindVertexArray(0);

            vol_gen_->InsertParticles(static_cast<unsigned int>(particles.GetCount()), global_radius, vertex_array_);
        }

        vol_gen_->EndInsertion();
        vol_gen_->RecreateMipmap();
    }

    // set texture handle
    texture_handle = vol_gen_->GetVolumeTextureHandle();

    return true;
}

bool VoxelGenerator::dummyCallback(core::Call& call) {
    return true;
}

bool VoxelGenerator::fillVAO(
    const MultiParticleDataCall::Particles& parts, GLuint vert_buf, const void* vert_ptr, bool create_buffer_data) {

    GLuint vert_attrib_loc = glGetAttribLocation(sphere_prgm_->getHandle(), "inPosition");

    const void* vertex_ptr = vert_ptr;
    if (create_buffer_data) {
        vertex_ptr = nullptr;
    }

    unsigned int part_count = static_cast<unsigned int>(parts.GetCount());

    // radius and position
    glBindBuffer(GL_ARRAY_BUFFER, vert_buf); //VBO

    switch (parts.GetVertexDataType()) {
    case MultiParticleDataCall::Particles::VERTDATA_NONE:
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        if (create_buffer_data) {
            glBufferData(GL_ARRAY_BUFFER,
                part_count * (std::max)(parts.GetVertexDataStride(), static_cast<unsigned int>(3 * sizeof(float))),
                parts.GetVertexData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(vert_attrib_loc);
        glVertexAttribPointer(vert_attrib_loc, 3, GL_FLOAT, GL_FALSE, parts.GetVertexDataStride(), vertex_ptr);
        break;
    case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
        if (create_buffer_data) {
            glBufferData(GL_ARRAY_BUFFER,
                part_count * (std::max)(parts.GetVertexDataStride(), static_cast<unsigned int>(4 * sizeof(float))),
                parts.GetVertexData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(vert_attrib_loc);
        glVertexAttribPointer(vert_attrib_loc, 4, GL_FLOAT, GL_FALSE, parts.GetVertexDataStride(), vertex_ptr);
        break;
    case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
        if (create_buffer_data) {
            glBufferData(GL_ARRAY_BUFFER,
                part_count * (std::max)(parts.GetVertexDataStride(), static_cast<unsigned int>(3 * sizeof(double))),
                parts.GetVertexData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(vert_attrib_loc);
        glVertexAttribPointer(vert_attrib_loc, 3, GL_DOUBLE, GL_FALSE, parts.GetVertexDataStride(), vertex_ptr);
        break;
    case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
        if (create_buffer_data) {
            glBufferData(GL_ARRAY_BUFFER,
                part_count * (std::max)(parts.GetVertexDataStride(), static_cast<unsigned int>(3 * sizeof(short))),
                parts.GetVertexData(), GL_STATIC_DRAW);
        }
        glEnableVertexAttribArray(vert_attrib_loc);
        glVertexAttribPointer(vert_attrib_loc, 3, GL_SHORT, GL_FALSE, parts.GetVertexDataStride(), vertex_ptr);
        break;
    default:
        break;
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return true;
}
