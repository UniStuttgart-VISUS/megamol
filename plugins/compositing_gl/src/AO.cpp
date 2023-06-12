#include "AO.h"

#include "OpenGL_Context.h"
#include "mmcore/param/IntParam.h"
#include "compositing_gl/CompositingCalls.h"

using namespace megamol::core;
using namespace megamol::geocalls;

#define AO_DIR_UBO_BINDING_POINT 0


megamol::compositing_gl::AO::AO(void)
        : mmstd_gl::Renderer3DModuleGL()
        , output_tex_slot_("AmbientOcclusionTexture", "Slot for requesting ambient occlusion texture.")
        , voxels_tex_slot_("getVoxelData", "Connects to the voxel data source")
        , vol_size_slot_("volumeSize", "Longest volume edge")
        , texture_handle()
        , voxel_handle()
        , vertex_array_()
        , shader_options_flags_(nullptr)
        , lighting_prgm_()
        , vbo_()
        , cur_vp_width_(-1)
        , cur_vp_height_(-1)
        , cur_mvp_inv_()
        , cur_cam_pos_()
        , ao_dir_ubo_(nullptr)
        , ao_offset_slot_("offset", "Offset from Surface")
        , ao_strength_slot_("strength", "Strength")
        , ao_cone_apex_slot_("apex", "Cone Apex Angle")
        , ao_cone_length_slot_("coneLength", "Cone length")
        , enable_lighting_slot_("enableLighting", "Enable Lighting")
        , use_hp_textures_slot_("highPrecisionTexture", "Use high precision textures")
        , normals_tex_slot_("NormalTexture", "Connects the normals render target texture")
        , depth_tex_slot_("DepthTexture", "Connects the depth render target texture")
        , color_tex_slot_("ColorTexture", "Connects the color render target texture")
        , camera_slot_("Camera", "Connects a (copy of) camera state")
        , depth_tex()
        , normal_tex()
        , color_tex()
        {

    // CallTexture2D
    //this->output_tex_slot_.SetCallback(CallTexture2D::ClassName(), "GetData", &AO::getDataCallback);
    //this->output_tex_slot_.SetCallback(CallTexture2D::ClassName(), "GetMetaData", &AO::getMetadataCallback);
    //this->MakeSlotAvailable(&this->output_tex_slot_);

    // VolumetricDataCall slot (get voxel texture)
    this->voxels_tex_slot_.SetCompatibleCall<VolumetricDataCallDescription>();
    this->voxels_tex_slot_.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->voxels_tex_slot_);

    //this->vol_size_slot_ << (new core::param::IntParam(256, 1, 1024));
    //this->MakeSlotAvailable(&this->vol_size_slot_);

    this->enable_lighting_slot_ << (new param::BoolParam(false)); // TODO necessary?
    this->MakeSlotAvailable(&this->enable_lighting_slot_);

    this->use_hp_textures_slot_ << (new param::BoolParam(false));
    this->MakeSlotAvailable(&this->use_hp_textures_slot_);

    this->ao_cone_apex_slot_ << (new param::FloatParam(50.0f, 1.0f, 90.0f));
    this->MakeSlotAvailable(&this->ao_cone_apex_slot_);

    this->ao_offset_slot_ << (new param::FloatParam(0.01f, 0.0f, 0.2f));
    this->MakeSlotAvailable(&this->ao_offset_slot_);

    this->ao_strength_slot_ << (new param::FloatParam(1.0f, 0.1f, 20.0f));
    this->MakeSlotAvailable(&this->ao_strength_slot_);

    this->ao_cone_length_slot_ << (new param::FloatParam(0.8f, 0.01f, 1.0f));
    this->MakeSlotAvailable(&this->ao_cone_length_slot_);

    this->color_tex_slot_.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->color_tex_slot_);

    this->normals_tex_slot_.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->normals_tex_slot_);

    this->depth_tex_slot_.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->depth_tex_slot_);

    this->camera_slot_.SetCompatibleCall<CallCameraDescription>();
    this->MakeSlotAvailable(&this->camera_slot_);
}



megamol::compositing_gl::AO::~AO(void) {

    this->release();
}

void megamol::compositing_gl::AO::release(void) {}

bool megamol::compositing_gl::AO::create(void){
    return recreateResources();
}

bool megamol::compositing_gl::AO::recreateResources(void) {

    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>(); //TODO necessary?

    // Check for flag storage availability and get specific shader snippet
    // TODO: test flags!
    // create shader programs
    auto const shader_options = core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());
    shader_options_flags_ = std::make_unique<msf::ShaderFactoryOptionsOpenGL>(shader_options);

     // Create the deferred shader
    auto lighting_so = shader_options;

    bool enable_lighting = this->enable_lighting_slot_.Param<param::BoolParam>()->Value(); // TODO redo this when enable_lighting_slot_ changed
    if (enable_lighting) {
        lighting_so.addDefinition("ENABLE_LIGHTING");
    }

    float apex = this->ao_cone_apex_slot_.Param<param::FloatParam>()->Value();
    std::vector<glm::vec4> directions;
    this->generate3ConeDirections(directions, apex * static_cast<float>(M_PI) / 180.0f);
    lighting_so.addDefinition("NUM_CONEDIRS", std::to_string(directions.size()));

    std::vector<float> dummy = {0};
    ao_dir_ubo_ = std::make_unique<glowl::BufferObject>(GL_UNIFORM_BUFFER, dummy);
    ao_dir_ubo_->rebuffer(directions);

    lighting_prgm_.reset();
    lighting_prgm_ = core::utility::make_glowl_shader("sphere_mdao_deferred", lighting_so,
        "moldyn_gl/sphere_renderer/sphere_mdao_deferred.vert.glsl",
        "moldyn_gl/sphere_renderer/sphere_mdao_deferred.frag.glsl");

    // TODO glowl implementation of GLSLprogram misses this functionality
    auto ubo_idx = glGetUniformBlockIndex(lighting_prgm_->getHandle(), "cone_buffer");
    glUniformBlockBinding(lighting_prgm_->getHandle(), ubo_idx, (GLuint)AO_DIR_UBO_BINDING_POINT);

    return true;
}

bool megamol::compositing_gl::AO::GetExtents(mmstd_gl::CallRender3DGL& call) {
    auto cr = &call;
    // TODO Set bounding box through volume bounding box?
    cur_clip_box_ = cr->AccessBoundingBoxes().ClipBox();
    return true;
}


bool megamol::compositing_gl::AO::Render(mmstd_gl::CallRender3DGL& call) {
    auto callNormal = normals_tex_slot_.CallAs<CallTexture2D>();
    auto callDepth = depth_tex_slot_.CallAs<CallTexture2D>();
    auto callColor = color_tex_slot_.CallAs<CallTexture2D>();
    auto callCamera = camera_slot_.CallAs<CallCamera>();
    auto callVoxel = voxels_tex_slot_.CallAs<VolumetricDataCall>();

    if (callNormal == NULL)
        return false;
    if (callDepth == NULL)
        return false;
    if (callColor == NULL)
        return false;
    if (callCamera == NULL)
        return false;
    if (callVoxel == NULL)
        return false;


    if (!(*callNormal)(0))
        return false;

    if (!(*callDepth)(0))
        return false;

    if (!(*callColor)(0))
        return false;

    if (!(*callCamera)(0))
        return false;


    bool normalUpdate = callNormal->hasUpdate();
    bool depthUpdate = callDepth->hasUpdate();
    bool colorUpdate = callColor->hasUpdate();
    bool cameraUpdate = callCamera->hasUpdate();

    if (normalUpdate || depthUpdate || colorUpdate || cameraUpdate) {

        // get textures
        normal_tex = callNormal->getData();
        depth_tex = callDepth->getData();
        color_tex = callColor->getData();

        // update volume texture
        updateVolumeData(call.Time());

        // obtain camera information
        core::view::Camera cam = callCamera->getData();
        glm::mat4 viewMx = cam.getViewMatrix();
        glm::mat4 projMx = cam.getProjectionMatrix();
        cur_mvp_inv_ = glm::inverse(projMx * viewMx);
        cur_cam_pos_ = cam.getPose().position;

        // fbo info
        auto fbo = call.GetFramebuffer();
        cur_vp_width_ = fbo->getWidth();
        cur_vp_height_ = fbo->getHeight();

        // update shader
        recreateResources();

        renderAmbientOcclusion();
    }
    return true;
}

void megamol::compositing_gl::AO::generate3ConeDirections(std::vector<glm::vec4>& directions, float apex) {

    directions.clear();

    float edge_length = 2.0f * tan(0.5f * apex);
    float height = sqrt(1.0f - edge_length * edge_length / 12.0f);
    float radius = sqrt(3.0f) / 3.0f * edge_length;

    for (int i = 0; i < 3; i++) {
        float angle = static_cast<float>(i) / 3.0f * 2.0f * static_cast<float>(M_PI);

        glm::vec3 center(cos(angle) * radius, height, sin(angle) * radius);
        center = glm::normalize(center);
        directions.push_back(glm::vec4(center.x, center.y, center.z, edge_length));
    }
}


bool megamol::compositing_gl::AO::updateVolumeData(const unsigned int frameID) {
    VolumetricDataCall* c_voxel = this->voxels_tex_slot_.CallAs<VolumetricDataCall>();

    if (c_voxel != nullptr) {
        c_voxel->SetFrameID(frameID);
        do {
            if (!(*c_voxel)(VolumetricDataCall::IDX_GET_EXTENTS)) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "SphereRenderer: could not get all extents (VolumetricDataCall)");
                return false;
            }
            if (!(*c_voxel)(VolumetricDataCall::IDX_GET_METADATA)) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "SphereRenderer: could not get metadata (VolumetricDataCall)");
                return false;
            }
            if (!(*c_voxel)(VolumetricDataCall::IDX_GET_DATA)) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "SphereRenderer: could not get data (VolumetricDataCall)");
                return false;
            }

            // TODO get datahash and frameId
        } while (c_voxel->FrameID() != frameID);
    }

    return true;
}

void megamol::compositing_gl::AO::renderAmbientOcclusion(){

    
    //glEnable(GL_TEXTURE_2D);
    //glEnable(GL_TEXTURE_3D);

    bool enable_lighting = this->enable_lighting_slot_.Param<param::BoolParam>()->Value();
    bool high_precision = this->use_hp_textures_slot_.Param<param::BoolParam>()->Value();

    glActiveTexture(GL_TEXTURE2);
    depth_tex->bindTexture();
    glActiveTexture(GL_TEXTURE1);
    normal_tex->bindTexture();
    glActiveTexture(GL_TEXTURE0);
    color_tex->bindTexture();

    // voxel texture
    VolumetricDataCall* c_voxel = this->voxels_tex_slot_.CallAs<VolumetricDataCall>();
    int vol_size = c_voxel->GetMetadata()->Resolution[0]; // TODO access correct value
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_3D, c_voxel->GetVRAMData());
    glActiveTexture(GL_TEXTURE0);


    // shader
    this->lighting_prgm_->use();

    ao_dir_ubo_->bind((GLuint)AO_DIR_UBO_BINDING_POINT);

    this->lighting_prgm_->setUniform("inColorTex", static_cast<int>(0));
    this->lighting_prgm_->setUniform("inNormalsTex", static_cast<int>(1));
    this->lighting_prgm_->setUniform("inDepthTex", static_cast<int>(2));
    this->lighting_prgm_->setUniform("inDensityTex", static_cast<int>(3));
    

    this->lighting_prgm_->setUniform("inWidth", static_cast<float>(cur_vp_width_));
    this->lighting_prgm_->setUniform("inHeight", static_cast<float>(cur_vp_height_));
    glUniformMatrix4fv(this->lighting_prgm_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
    this->lighting_prgm_->setUniform("inUseHighPrecision", high_precision);
    if (enable_lighting) {
        this->lighting_prgm_->setUniform("inObjLightDir", glm::vec3(1.0, 1.0, 0.0));//this->cur_light_dir_)); // TODO
        this->lighting_prgm_->setUniform("inObjCamPos", cur_cam_pos_);
    }
    this->lighting_prgm_->setUniform("inAOOffset", this->ao_offset_slot_.Param<param::FloatParam>()->Value());
    this->lighting_prgm_->setUniform("inAOStrength", this->ao_strength_slot_.Param<param::FloatParam>()->Value());
    this->lighting_prgm_->setUniform("inAOConeLength", this->ao_cone_length_slot_.Param<param::FloatParam>()->Value());

    // calculate amb cone constants
    vislib::math::Dimension<float, 3> dims = this->cur_clip_box_.GetSize();
    // Calculate the extensions of the volume by using the specified number of voxels for the longest edge
    float longest_edge = this->cur_clip_box_.LongestEdge();
    dims.Scale(static_cast<float>(vol_size) / longest_edge);
    // The X size must be a multiple of 4, so we might have to correct that a little
    dims.SetWidth(ceil(dims.GetWidth() / 4.0f) * 4.0f);
    dims.SetHeight(ceil(dims.GetHeight()));
    dims.SetDepth(ceil(dims.GetDepth()));
    float amb_cone_constant_0 = std::min(dims.Width(), std::min(dims.Height(), dims.Depth()));
    float amb_cone_constant_1 = ceil(std::log2(static_cast<float>(vol_size))) - 1.0f;

    this->lighting_prgm_->setUniform("inAmbVolShortestEdge", amb_cone_constant_0);
    this->lighting_prgm_->setUniform("inAmbVolMaxLod", amb_cone_constant_1);
    glm::vec3 cur_clip_box_coords = glm::vec3(this->cur_clip_box_.GetLeftBottomBack().GetX(),
        this->cur_clip_box_.GetLeftBottomBack().GetY(), this->cur_clip_box_.GetLeftBottomBack().GetZ());
    this->lighting_prgm_->setUniform("inBoundsMin", cur_clip_box_coords);
    glm::vec3 cur_clip_box_size = glm::vec3(this->cur_clip_box_.GetSize().GetWidth(),
        this->cur_clip_box_.GetSize().GetHeight(), this->cur_clip_box_.GetSize().GetDepth());
    this->lighting_prgm_->setUniform("inBoundsSize", cur_clip_box_size);

    // Draw screen filling 'quad' (2 triangle, front facing: CCW)
    std::vector<GLfloat> vertices = {-1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f};
    GLuint vert_attrib_loc = glGetAttribLocation(this->lighting_prgm_->getHandle(), "inPosition");
    glEnableVertexAttribArray(vert_attrib_loc);
    glVertexAttribPointer(vert_attrib_loc, 2, GL_FLOAT, GL_TRUE, 0, vertices.data());
    glDrawArrays(GL_TRIANGLES, static_cast<GLint>(0), static_cast<GLsizei>(vertices.size() / 2));
    glDisableVertexAttribArray(vert_attrib_loc);

    glUseProgram(0); // this->lighting_prgm_.Disable();

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_3D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    //glDisable(GL_TEXTURE_2D);
    //glDisable(GL_TEXTURE_3D);


    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        // Process/log the error.
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("AO: error" + err);
    }
}
