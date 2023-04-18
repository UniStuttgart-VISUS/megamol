#include "AO.h"

#include "OpenGL_Context.h"
#include "mmcore/param/IntParam.h"

using namespace megamol::moldyn_gl::rendering;
using namespace megamol::geocalls;

AO::AO(void)
        : mmstd_gl::ModuleGL()
        , output_tex_slot_("generateVoxels", "Slot for requesting voxel generation.")
        , voxels_tex_slot_("getParticleData", "Connects to the data source")
        , vol_size_slot_("volumeSize", "Longest volume edge")
        , texture_handle()
        , voxel_handle()
        , vol_gen_(nullptr)
        , vertex_array_()
        , shader_options_flags_(nullptr)
        , sphere_prgm_()
        , vbo_() {

    // CallTexture2D
    this->output_tex_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA), &AO::getDataCallback);
    this->output_tex_slot_.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA), &AO::getMetadataCallback);

    this->MakeSlotAvailable(&this->output_tex_slot_);

    // MultiParticleDataCall slot
    this->voxels_tex_slot_.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->voxels_tex_slot_.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->voxels_tex_slot_);

    this->vol_size_slot_ << (new core::param::IntParam(256, 1, 1024));
    this->MakeSlotAvailable(&this->vol_size_slot_);
}

AO::~AO(void) {

    this->release();
}

void AO::release(void) {
}

bool AO::create(void) {
    return true;
}


bool AO::getMetadataCallback(core::Call& call) {
    return true;
}


bool AO::getDataCallback(core::Call& call) {

    return true;
}

// AO (sphere renderer)
// shaders
// rebuild working data (particle data, volume data)
// render particle geometry (or get from sphere renderer)  sphere_geometry_prgm_ or sphere_prgm_
// render deferred pass


// render deferred pass:
// need depth, normals, color, voxel texture, shader (sphere_mdao_deferred)







//void SphereRenderer::renderDeferredPass(mmstd_gl::CallRender3DGL& call) {
//
//    bool enable_lighting = this->enable_lighting_slot_.Param<param::BoolParam>()->Value();
//    bool high_precision = this->use_hp_textures_slot_.Param<param::BoolParam>()->Value();
//
//    glActiveTexture(GL_TEXTURE2);
//    glBindTexture(GL_TEXTURE_2D, this->g_buffer_.depth);
//    glActiveTexture(GL_TEXTURE1);
//    glBindTexture(GL_TEXTURE_2D, this->g_buffer_.normals);
//    glActiveTexture(GL_TEXTURE0);
//    glBindTexture(GL_TEXTURE_2D, this->g_buffer_.color);
//
//    // voxel texture
//    VolumetricDataCall* c_voxel = this->get_voxels_.CallAs<VolumetricDataCall>();
//    if (c_voxel != nullptr) {
//        if (!(*c_voxel)(VolumetricDataCall::IDX_GET_METADATA)) {
//            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
//                "SphereRenderer: could not get metadata (VolumetricDataCall)");
//        } else {
//            // get volume size from metadata
//            int new_vol_size = c_voxel->GetMetadata()->Resolution[0]; //TODO access correct value
//            if (new_vol_size != vol_size_) {
//                vol_size_ = new_vol_size;
//                vol_size_changed_ = true;
//            }
//            glActiveTexture(GL_TEXTURE3);
//            glBindTexture(GL_TEXTURE_3D, c_voxel->GetVRAMData());
//            glActiveTexture(GL_TEXTURE0);
//        }
//    }
//
//    this->lighting_prgm_->use();
//
//    ao_dir_ubo_->bind((GLuint)AO_DIR_UBO_BINDING_POINT);
//
//    this->lighting_prgm_->setUniform("inColorTex", static_cast<int>(0));
//    this->lighting_prgm_->setUniform("inNormalsTex", static_cast<int>(1));
//    this->lighting_prgm_->setUniform("inDepthTex", static_cast<int>(2));
//    this->lighting_prgm_->setUniform("inDensityTex", static_cast<int>(3));
//
//    this->lighting_prgm_->setUniform("inWidth", static_cast<float>(this->cur_vp_width_));
//    this->lighting_prgm_->setUniform("inHeight", static_cast<float>(this->cur_vp_height_));
//    glUniformMatrix4fv(
//        this->lighting_prgm_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(this->cur_mvp_inv_));
//    this->lighting_prgm_->setUniform("inUseHighPrecision", high_precision);
//    if (enable_lighting) {
//        this->lighting_prgm_->setUniform("inObjLightDir", glm::vec3(this->cur_light_dir_));
//        this->lighting_prgm_->setUniform("inObjCamPos", glm::vec3(this->cur_cam_pos_));
//    }
//    this->lighting_prgm_->setUniform("inAOOffset", this->ao_offset_slot_.Param<param::FloatParam>()->Value());
//    this->lighting_prgm_->setUniform("inAOStrength", this->ao_strength_slot_.Param<param::FloatParam>()->Value());
//    this->lighting_prgm_->setUniform("inAOConeLength", this->ao_cone_length_slot_.Param<param::FloatParam>()->Value());
//    this->lighting_prgm_->setUniform("inAmbVolShortestEdge", this->amb_cone_constants_[0]);
//    this->lighting_prgm_->setUniform("inAmbVolMaxLod", this->amb_cone_constants_[1]);
//    glm::vec3 cur_clip_box_coords = glm::vec3(this->cur_clip_box_.GetLeftBottomBack().GetX(),
//        this->cur_clip_box_.GetLeftBottomBack().GetY(), this->cur_clip_box_.GetLeftBottomBack().GetZ());
//    this->lighting_prgm_->setUniform("inBoundsMin", cur_clip_box_coords);
//    glm::vec3 cur_clip_box_size = glm::vec3(this->cur_clip_box_.GetSize().GetWidth(),
//        this->cur_clip_box_.GetSize().GetHeight(), this->cur_clip_box_.GetSize().GetDepth());
//    this->lighting_prgm_->setUniform("inBoundsSize", cur_clip_box_size);
//
//    // Draw screen filling 'quad' (2 triangle, front facing: CCW)
//    std::vector<GLfloat> vertices = {-1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f};
//    GLuint vert_attrib_loc = glGetAttribLocation(this->lighting_prgm_->getHandle(), "inPosition");
//    glEnableVertexAttribArray(vert_attrib_loc);
//    glVertexAttribPointer(vert_attrib_loc, 2, GL_FLOAT, GL_TRUE, 0, vertices.data());
//    glDrawArrays(GL_TRIANGLES, static_cast<GLint>(0), static_cast<GLsizei>(vertices.size() / 2));
//    glDisableVertexAttribArray(vert_attrib_loc);
//
//    glUseProgram(0); // this->lighting_prgm_.Disable();
//
//    glActiveTexture(GL_TEXTURE1);
//    glBindTexture(GL_TEXTURE_2D, 0);
//    glActiveTexture(GL_TEXTURE2);
//    glBindTexture(GL_TEXTURE_2D, 0);
//    glActiveTexture(GL_TEXTURE3);
//    glBindTexture(GL_TEXTURE_3D, 0);
//    glActiveTexture(GL_TEXTURE0);
//    glBindTexture(GL_TEXTURE_2D, 0);
//
//    glDisable(GL_TEXTURE_2D);
//    glDisable(GL_TEXTURE_3D);
//}
