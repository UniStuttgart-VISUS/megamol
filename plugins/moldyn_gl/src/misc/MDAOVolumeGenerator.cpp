
#include "misc/MDAOVolumeGenerator.h"
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/utility/log/Log.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <GL/glu.h>


using namespace megamol::moldyn_gl::misc;


#define checkGLError                                                                                   \
    {                                                                                                  \
        GLenum errCode = glGetError();                                                                 \
        if (errCode != GL_NO_ERROR)                                                                    \
            std::cout << "Error in line " << __LINE__ << ": " << gluErrorString(errCode) << std::endl; \
    }


MDAOVolumeGenerator::MDAOVolumeGenerator()
        : shader_options_(nullptr)
        , fbo_handle_(0)
        , volume_handle_(0)
        , data_version_(0) {
    clear_buffer_ = nullptr;
}


MDAOVolumeGenerator::~MDAOVolumeGenerator() {
    glDeleteTextures(1, &volume_handle_);

    if (clear_buffer_ != nullptr)
        delete[] clear_buffer_;
}

void MDAOVolumeGenerator::SetShaderSourceFactory(msf::ShaderFactoryOptionsOpenGL* so) {
    this->shader_options_ = so;
}

GLuint MDAOVolumeGenerator::GetVolumeTextureHandle() {
    return volume_handle_;
}


bool MDAOVolumeGenerator::Init(frontend_resources::OpenGL_Context const& ogl_ctx) {
    // Generate and initialize the volume texture
    glGenTextures(1, &this->volume_handle_);
    glBindTexture(GL_TEXTURE_3D, this->volume_handle_);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

    // Generate and initialize the frame buffer
    glGenFramebuffers(1, &fbo_handle_);

    GLint prev_fbo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_handle_);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, this->volume_handle_, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);

    // Check if we can use modern features
    compute_available_ = ogl_ctx.isExtAvailable("GL_ARB_compute_shader");
    clear_available_ = ogl_ctx.isExtAvailable("GL_ARB_clear_texture");

    std::stringstream outmsg;
    outmsg << "[MDAOVolumeGenerator] Voxelization Features enabled: Compute Shader " << compute_available_
           << ", Clear Texture " << clear_available_ << std::endl;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(outmsg.str().c_str());

    // create shader programs
    if (compute_available_) {
        // Try to initialize the compute shader
        try {
            mipmap_prgm_ = core::utility::make_glowl_shader(
                "mipmap", *shader_options_, "moldyn_gl/mdao_volume_generator/mdao_mipmap_compute_main.comp.glsl");
        } catch (std::exception& e) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to compile mdao volume generator shader: %s. [%s, %s, line %d]\n",
                std::string(e.what()).c_str(), __FILE__, __FUNCTION__, __LINE__);

            return false;
        }
    }

    // Initialize our shader
    try {
        // Try to make the volume shader
        volume_prgm_ = core::utility::make_glowl_shader("volume", *shader_options_,
            "moldyn_gl/mdao_volume_generator/mdao_volume.vert.glsl",
            "moldyn_gl/mdao_volume_generator/mdao_volume.geom.glsl",
            "moldyn_gl/mdao_volume_generator/mdao_volume.frag.glsl");

    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to compile mdao volume generator shader: %s. [%s, %s, line %d]\n", std::string(e.what()).c_str(),
            __FILE__, __FUNCTION__, __LINE__);

        return false;
    }

    return true;
}


void MDAOVolumeGenerator::SetResolution(float res_x, float res_y, float res_z) {
    if (volume_res_.Width() == res_x && volume_res_.Height() == res_y && volume_res_.Depth() == res_z)
        return;

    if (!clear_available_) {
        if (clear_buffer_ != nullptr)
            delete[] clear_buffer_;
        clear_buffer_ = new unsigned char[static_cast<unsigned int>(ceil(res_x * res_y * res_z))];
        memset(clear_buffer_, 0, static_cast<unsigned int>(res_x * res_y * res_z));
    }

    volume_res_.Set(res_x, res_y, res_z);
    glBindTexture(GL_TEXTURE_3D, this->volume_handle_);
    checkGLError;
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, static_cast<GLsizei>(res_x), static_cast<GLsizei>(res_y),
        static_cast<GLsizei>(res_z), 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    checkGLError glGenerateMipmap(GL_TEXTURE_3D);
}


void MDAOVolumeGenerator::StartInsertion(const vislib::math::Cuboid<float>& obb, const glm::vec4& clip_dat_) {
    this->clip_dat_ = clip_dat_;

    this->bounds_size_inverse_ = obb.GetSize();
    for (int i = 0; i < 3; ++i) {
        this->bounds_size_inverse_[i] = 1.0f / this->bounds_size_inverse_[i];
    }

    auto bmin = obb.GetLeftBottomBack();
    this->bounds_min_ = glm::vec3(bmin.GetX(), bmin.GetY(), bmin.GetZ());

    // Save previous OpenGL state
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    checkGLError;
    glEnable(GL_BLEND);
    checkGLError;
    glBlendFunc(GL_ONE, GL_ONE);
    checkGLError;
    glGetIntegerv(GL_VIEWPORT, viewport_);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo_);

    glBindTexture(GL_TEXTURE_3D, this->volume_handle_);
    checkGLError;
    glBindFramebuffer(GL_FRAMEBUFFER, this->fbo_handle_);
    checkGLError;

    // Set viewport_ size for one slice
    glViewport(0, 0, static_cast<GLsizei>(volume_res_.GetWidth()), static_cast<GLsizei>(volume_res_.GetHeight()));

    volume_prgm_->use();
    checkGLError;

    volume_prgm_->setUniform("inBoundsMin", this->bounds_min_);
    volume_prgm_->setUniform(
        "inBoundsSizeInverse", glm::vec3(this->bounds_size_inverse_.GetWidth(), this->bounds_size_inverse_.GetHeight(),
                                   this->bounds_size_inverse_.GetDepth()));
    volume_prgm_->setUniform("inVolumeSize",
        glm::vec3(this->volume_res_.GetWidth(), this->volume_res_.GetHeight(), this->volume_res_.GetDepth()));
    volume_prgm_->setUniform("clip_dat_", clip_dat_);
}


void MDAOVolumeGenerator::ClearVolume() {
    if (clear_available_) {
        unsigned char clearData = 0;
        glClearTexImage(this->volume_handle_, 0, GL_RED, GL_UNSIGNED_BYTE, &clearData);
        checkGLError;
        return;
    }

    if (clear_buffer_ == nullptr)
        return;

    glBindTexture(GL_TEXTURE_3D, this->volume_handle_);
    checkGLError;
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, static_cast<GLsizei>(volume_res_.Width()),
        static_cast<GLsizei>(volume_res_.Height()), static_cast<GLsizei>(volume_res_.Depth()), 0, GL_RED,
        GL_UNSIGNED_BYTE, clear_buffer_);
    checkGLError;
    glBindTexture(GL_TEXTURE_3D, 0);
    checkGLError;
}


void MDAOVolumeGenerator::EndInsertion() {
    glUseProgram(0); // volume_prgm_.Disable();
    checkGLError;

    glBindVertexArray(0);

    // Reset previous OpenGL state
    glViewport(viewport_[0], viewport_[1], viewport_[2], viewport_[3]);
    glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo_);
    glBindTexture(GL_TEXTURE_3D, 0);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_BLEND);

    //glDisable(GL_VERTEX_PROGRAM_POINT_SIZE); /// ! Do not disable, still required in sphere renderer !

    ++data_version_;

    //glBindTexture(GL_TEXTURE_3D, this->volume_handle_);
    //std::vector<char> rawdata(volume_res_.Width() * volume_res_.Height() * volume_res_.Depth());
    //glGetTexImage(GL_TEXTURE_3D, 0, GL_RED, GL_UNSIGNED_BYTE, rawdata.data());
    //std::ofstream outfile("volume.raw", std::ios::binary);
    //outfile.write(rawdata.data(), rawdata.size());
    //outfile.close();
    //std::cout<<"Wrote volume: "<<volume_res_.Width()<<" x "<<volume_res_.Height()<<" x "<<volume_res_.Depth()<<std::endl;
    //exit(123);
}


void MDAOVolumeGenerator::InsertParticles(unsigned int count, float globalRadius, GLuint vertexArray) {
    volume_prgm_->setUniform("inGlobalRadius", globalRadius);
    checkGLError;

    glBindVertexArray(vertexArray);
    checkGLError;
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(count));
    checkGLError;
}


void MDAOVolumeGenerator::RecreateMipmap() {

    if (!compute_available_) {
        glBindTexture(GL_TEXTURE_3D, this->volume_handle_);
        glGenerateMipmap(GL_TEXTURE_3D);
        glBindTexture(GL_TEXTURE_3D, 0);
        return;
    }

    // int res = volume_res_.Width();
    vislib::math::Dimension<float, 3> res = volume_res_;
    int level = 0;

    mipmap_prgm_->use();
    while (res.Width() >= 1.0f || res.Height() >= 1.0f || res.Depth() >= 1.0f) {
        res.Scale(0.5f);
        // Bind input image
        glBindImageTexture(0, this->volume_handle_, level, GL_TRUE, 0, GL_READ_ONLY, GL_R8);
        checkGLError;
        // Bind output image
        glBindImageTexture(1, this->volume_handle_, level + 1, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);
        checkGLError;

        mipmap_prgm_->setUniform("inputImage", 0);
        mipmap_prgm_->setUniform("outputImage", 1);
        glDispatchCompute(static_cast<GLuint>(ceil(res.Width() / 64.0f)), static_cast<GLuint>(ceil(res.Height())),
            static_cast<GLuint>(ceil(res.Depth())));
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        level++;
    };

    glUseProgram(0); // mipmap_prgm_.Disable();
}


const vislib::math::Dimension<float, 3>& MDAOVolumeGenerator::GetExtents() {
    return volume_res_;
}


unsigned int MDAOVolumeGenerator::GetDataVersion() {
    return data_version_;
}
