#include "AO.h"

#include "OpenGL_Context.h"
#include "mmcore/param/IntParam.h"
#include "compositing_gl/CompositingCalls.h"

using namespace megamol::core;
using namespace megamol::geocalls;

#define AO_DIR_UBO_BINDING_POINT 0


megamol::compositing_gl::AO::AO(void)
        : mmstd_gl::ModuleGL()
        , output_tex_slot_("AmbientOcclusionTexture", "Slot for requesting ambient occlusion texture.")
        , voxels_tex_slot_("getVoxelData", "Connects to the voxel data source")
        , vol_size_slot_("volumeSize", "Longest volume edge")
        , texture_handle()
        , voxel_handle()
        , vertex_array_()
        , shader_options_flags_(nullptr)
        , lighting_prgm_()
        , vbo_()
        , ao_dir_ubo_(nullptr)
        , ao_offset_slot_("offset", "Offset from Surface")
        , ao_strength_slot_("strength", "Strength")
        , ao_cone_apex_slot_("apex", "Cone Apex Angle")
        , ao_cone_length_slot_("coneLength", "Cone length")
        , enable_lighting_slot_("enableLighting", "Enable Lighting")
        , depth_tex()
        , normal_tex()
        , color_tex()
        {

    // CallTexture2D
    this->output_tex_slot_.SetCallback(CallTexture2D::ClassName(), "GetData", &AO::getDataCallback);
    this->output_tex_slot_.SetCallback(CallTexture2D::ClassName(), "GetMetaData", &AO::getMetadataCallback);
    this->MakeSlotAvailable(&this->output_tex_slot_);

    // VolumetricDataCall slot (get voxel texture)
    this->voxels_tex_slot_.SetCompatibleCall<VolumetricDataCallDescription>();
    this->voxels_tex_slot_.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->voxels_tex_slot_);

    //this->vol_size_slot_ << (new core::param::IntParam(256, 1, 1024));
    //this->MakeSlotAvailable(&this->vol_size_slot_);

    this->enable_lighting_slot_ << (new param::BoolParam(false)); // TODO necessary?
    this->MakeSlotAvailable(&this->enable_lighting_slot_);

    this->ao_cone_apex_slot_ << (new param::FloatParam(50.0f, 1.0f, 90.0f));
    this->MakeSlotAvailable(&this->ao_cone_apex_slot_);

    this->ao_offset_slot_ << (new param::FloatParam(0.01f, 0.0f, 0.2f));
    this->MakeSlotAvailable(&this->ao_offset_slot_);

    this->ao_strength_slot_ << (new param::FloatParam(1.0f, 0.1f, 20.0f));
    this->MakeSlotAvailable(&this->ao_strength_slot_);

    this->ao_cone_length_slot_ << (new param::FloatParam(0.8f, 0.01f, 1.0f));
    this->MakeSlotAvailable(&this->ao_cone_length_slot_);
}



megamol::compositing_gl::AO::~AO(void) {

    this->release();
}

void megamol::compositing_gl::AO::release(void) {}

bool megamol::compositing_gl::AO::create(void) {

    std::vector<float> dummy = {0};
    ao_dir_ubo_ = std::make_unique<glowl::BufferObject>(GL_UNIFORM_BUFFER, dummy);

    // Check for flag storage availability and get specific shader snippet
    // TODO: test flags!
    // create shader programs
    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());
    shader_options_flags_ = std::make_unique<msf::ShaderFactoryOptionsOpenGL>(shader_options);

     // Create the deferred shader
    auto lighting_so = shader_options;

    bool enable_lighting = this->enable_lighting_slot_.Param<param::BoolParam>()->Value();
    if (enable_lighting) {
        lighting_so.addDefinition("ENABLE_LIGHTING");
    }

    float apex = this->ao_cone_apex_slot_.Param<param::FloatParam>()->Value();
    std::vector<glm::vec4> directions;
    this->generate3ConeDirections(directions, apex * static_cast<float>(M_PI) / 180.0f);
    lighting_so.addDefinition("NUM_CONEDIRS", std::to_string(directions.size()));

    ao_dir_ubo_->rebuffer(directions);

    lighting_prgm_.reset();
    lighting_prgm_ = core::utility::make_glowl_shader("sphere_mdao_deferred", lighting_so,
        "moldyn_gl/sphere_renderer/sphere_mdao_deferred.vert.glsl",
        "moldyn_gl/sphere_renderer/sphere_mdao_deferred.frag.glsl");
    
    return true;
}

bool megamol::compositing_gl::AO::getMetadataCallback(core::Call& call) {
    return true;
}


bool megamol::compositing_gl::AO::getDataCallback(core::Call& call) {
    auto lhsTc = dynamic_cast<CallTexture2D*>(&call);
    //auto callNormal = normals_tex_slot_.CallAs<CallTexture2D>();
    //auto callDepth = depth_tex_slot_.CallAs<CallTexture2D>();
    //auto callCamera = camera_slot_.CallAs<CallCamera>();

    if (lhsTc == NULL)
        return false;

    
    //setupOutputTexture(depthTx2D, final_output_);

    renderAmbientOcclusion();

    //final_output_->bindImage(0, GL_WRITE_ONLY);

    // set data
    //lhsTc->setData(final_output_, version_);

    // TODO make texture available!

    return true;
}

// AO (sphere renderer)
// shaders
// rebuild working data (particle data, volume data)
// render particle geometry (or get from sphere renderer)  sphere_geometry_prgm_ or sphere_prgm_
// render deferred pass


// render deferred pass:
// need depth, normals, color, voxel texture, shader (sphere_mdao_deferred)

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


void megamol::compositing_gl::AO::renderAmbientOcclusion() {

//void SphereRenderer::renderDeferredPass(mmstd_gl::CallRender3DGL& call) {
//
    //bool enable_lighting = this->enable_lighting_slot_.Param<param::BoolParam>()->Value();
    //bool high_precision = this->use_hp_textures_slot_.Param<param::BoolParam>()->Value();

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, depth_tex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, normal_tex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, color_tex);

    // voxel texture
    VolumetricDataCall* c_voxel = this->voxels_tex_slot_.CallAs<VolumetricDataCall>();
    if (c_voxel != nullptr) {
        if (!(*c_voxel)(VolumetricDataCall::IDX_GET_METADATA)) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "AO: could not get metadata (VolumetricDataCall)");
        } else {
            // get volume size from metadata
            //int new_vol_size = c_voxel->GetMetadata()->Resolution[0]; //TODO access correct value
            //if (new_vol_size != vol_size_) {
            //    vol_size_ = new_vol_size;
            //    vol_size_changed_ = true;
            //}
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_3D, c_voxel->GetVRAMData());
            glActiveTexture(GL_TEXTURE0);
        }
    }

    this->lighting_prgm_->use();

    ao_dir_ubo_->bind((GLuint)AO_DIR_UBO_BINDING_POINT);

    this->lighting_prgm_->setUniform("inColorTex", static_cast<int>(0));
    this->lighting_prgm_->setUniform("inNormalsTex", static_cast<int>(1));
    this->lighting_prgm_->setUniform("inDepthTex", static_cast<int>(2));
    this->lighting_prgm_->setUniform("inDensityTex", static_cast<int>(3));

    /*this->lighting_prgm_->setUniform("inWidth", static_cast<float>(this->cur_vp_width_));
    this->lighting_prgm_->setUniform("inHeight", static_cast<float>(this->cur_vp_height_));
    glUniformMatrix4fv(
        this->lighting_prgm_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
    this->lighting_prgm_->setUniform("inUseHighPrecision", high_precision);
    if (enable_lighting) {
        this->lighting_prgm_->setUniform("inObjLightDir", glm::vec3(this->cur_light_dir_));
        this->lighting_prgm_->setUniform("inObjCamPos", glm::vec3(this->cur_cam_pos_));
    }*/
    this->lighting_prgm_->setUniform("inAOOffset", this->ao_offset_slot_.Param<param::FloatParam>()->Value());
    this->lighting_prgm_->setUniform("inAOStrength", this->ao_strength_slot_.Param<param::FloatParam>()->Value());
    this->lighting_prgm_->setUniform("inAOConeLength", this->ao_cone_length_slot_.Param<param::FloatParam>()->Value());
    /*this->lighting_prgm_->setUniform("inAmbVolShortestEdge", this->amb_cone_constants_[0]);
    this->lighting_prgm_->setUniform("inAmbVolMaxLod", this->amb_cone_constants_[1]);
    glm::vec3 cur_clip_box_coords = glm::vec3(this->cur_clip_box_.GetLeftBottomBack().GetX(),
        this->cur_clip_box_.GetLeftBottomBack().GetY(), this->cur_clip_box_.GetLeftBottomBack().GetZ());
    this->lighting_prgm_->setUniform("inBoundsMin", cur_clip_box_coords);
    glm::vec3 cur_clip_box_size = glm::vec3(this->cur_clip_box_.GetSize().GetWidth(),
        this->cur_clip_box_.GetSize().GetHeight(), this->cur_clip_box_.GetSize().GetDepth());
    this->lighting_prgm_->setUniform("inBoundsSize", cur_clip_box_size);*/

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

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_TEXTURE_3D);
}
