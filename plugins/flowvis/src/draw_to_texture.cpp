#include "stdafx.h"
#include "draw_to_texture.h"

#include "matrix_call.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/view/AbstractCallRender.h"
#include "mmcore/view/CallRender2D.h"

#include "compositing/CompositingCalls.h"

#include "vislib/sys/Log.h"

#include "glowl/FramebufferObject.hpp"

#include "glm/mat4x4.hpp"

#include <memory>

namespace megamol {
namespace flowvis {

draw_to_texture::draw_to_texture()
    : texture_slot("texture", "Target texture")
    , model_matrix_slot("model_matrix", "Model matrix defining the position of the texture in world space")
    , rendering_slot("rendering", "Input rendering")
    , width("width", "Texture width in pixel")
    , height("height", "Texture height in pixel")
    , keep_aspect_ratio(
          "keep_aspect_ratio", "Keep aspect ratio of the incoming bounding rectangle, ignoring the height parameter")
    , fbo(nullptr)
    , hash(-1) {

    // Connect output
    this->texture_slot.SetCallback(compositing::CallTexture2D::ClassName(), compositing::CallTexture2D::FunctionName(0),
        &draw_to_texture::get_data);
    this->texture_slot.SetCallback(compositing::CallTexture2D::ClassName(), compositing::CallTexture2D::FunctionName(1),
        &draw_to_texture::get_extent);
    this->MakeSlotAvailable(&this->texture_slot);

    this->model_matrix_slot.SetCallback(
        matrix_call::ClassName(), matrix_call::FunctionName(0), &draw_to_texture::get_matrix);
    this->MakeSlotAvailable(&this->model_matrix_slot);

    // Connect input
    this->rendering_slot.SetCompatibleCall<core::view::CallRender2DDescription>();
    this->MakeSlotAvailable(&this->rendering_slot);

    // Set up parameters
    this->width << new core::param::IntParam(4000);
    this->MakeSlotAvailable(&this->width);

    this->height << new core::param::IntParam(4000);
    this->MakeSlotAvailable(&this->height);

    this->keep_aspect_ratio << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->keep_aspect_ratio);
}

draw_to_texture::~draw_to_texture() { this->Release(); }

bool draw_to_texture::create() { return true; }

void draw_to_texture::release() {}

bool draw_to_texture::get_input_extent() {
    auto rc_ptr = this->rendering_slot.CallAs<core::view::CallRender2D>();

    if (rc_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("No input connected");
        return false;
    }

    auto& rc = *rc_ptr;

    if (!rc(core::view::AbstractCallRender::FnGetExtents)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting input rendering extent");
        return false;
    }

    this->bounding_rectangle = rc.GetBoundingBox();
}

bool draw_to_texture::get_data(core::Call& call) {
    auto rc_ptr = this->rendering_slot.CallAs<core::view::CallRender2D>();

    if (rc_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("No input connected");
        return false;
    }

    auto& rc = *rc_ptr;

    // Set dummy hash
    core::BasicMetaData meta_data;
    meta_data.m_data_hash = this->hash++;

    static_cast<compositing::CallTexture2D&>(call).setMetaData(meta_data);

    // Create texture if necessary
    const auto aspect_ratio = this->bounding_rectangle.AspectRatio();

    const auto width = this->width.Param<core::param::IntParam>()->Value();
    const auto height = this->keep_aspect_ratio.Param<core::param::BoolParam>()->Value()
                            ? static_cast<int>(this->width.Param<core::param::IntParam>()->Value() / aspect_ratio)
                            : this->height.Param<core::param::IntParam>()->Value();

    if (this->fbo == nullptr || this->width.IsDirty() || this->height.IsDirty()) {
        this->fbo = std::make_unique<glowl::FramebufferObject>(width, height);
        this->fbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

        this->width.ResetDirty();
        this->height.ResetDirty();
    }

    // Set view
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glm::mat4 view_mx(1.0f);
    view_mx[0][0] = 2.0f / this->bounding_rectangle.Width();
    view_mx[1][1] = 2.0f / this->bounding_rectangle.Height();
    view_mx[3][0] = -this->bounding_rectangle.Left() * view_mx[0][0] - 1.0f;
    view_mx[3][1] = -this->bounding_rectangle.Bottom() * view_mx[1][1] - 1.0f;
    glLoadMatrixf(glm::value_ptr(view_mx));

    // Render
    this->fbo->bind();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!rc(core::view::AbstractCallRender::FnRender)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting input rendering");
        return false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glViewport(vp[0], vp[1], vp[2], vp[3]);

    // Set output
    static_cast<compositing::CallTexture2D&>(call).setData(this->fbo->getColorAttachment(0));

    return true;
}

bool draw_to_texture::get_extent(core::Call&) { return get_input_extent(); }

bool draw_to_texture::get_matrix(core::Call& call) {
    if (!this->get_input_extent()) return false;

    // Assume texture quad to live in [0, 0] x [1, 1] and create model matrix to transform
    // this quad to the coordinates and extent of the bounding rectangle, setting z=0
    glm::mat4 scale(1.0f);
    scale[0][0] = this->bounding_rectangle.Width();
    scale[1][1] = this->bounding_rectangle.Height();

    glm::mat4 translate(1.0f);
    translate[3][0] = this->bounding_rectangle.Left();
    translate[3][1] = this->bounding_rectangle.Bottom();

    const glm::mat4 model = translate * scale;

    static_cast<matrix_call&>(call).set_matrix(model);

    return true;
}

} // namespace flowvis
} // namespace megamol
