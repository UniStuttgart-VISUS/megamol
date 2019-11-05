#include "stdafx.h"
#include "draw_to_texture.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/view/AbstractCallRender.h"
#include "mmcore/view/CallRender2D.h"

#include "compositing/CompositingCalls.h"

#include "vislib/Exception.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/sys/Log.h"

#include "glowl/Texture.hpp"
#include "glowl/Texture2D.hpp"

namespace megamol {
namespace flowvis {

draw_to_texture::draw_to_texture()
    : texture_slot("texture", "Target texture")
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
    if (this->fbo == nullptr) {
        const auto bounding_rectangle = rc.GetBoundingBox();
        const auto aspect_ratio = bounding_rectangle.AspectRatio();

        const auto width = this->width.Param<core::param::IntParam>()->Value();
        const auto height = this->keep_aspect_ratio.Param<core::param::BoolParam>()->Value()
                                ? static_cast<int>(this->width.Param<core::param::IntParam>()->Value() / aspect_ratio)
                                : this->height.Param<core::param::IntParam>()->Value();

        this->fbo = std::make_unique<glowl::FramebufferObject>(width, height);
        this->fbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    }

    // Render
    this->fbo->bind();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!rc(core::view::AbstractCallRender::FnRender)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting input rendering");
        return false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Set output
    static_cast<compositing::CallTexture2D&>(call).setData(this->fbo->getColorAttachment(0));

    return true;
}

bool draw_to_texture::get_extent(core::Call&) {
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

    return true;
}

} // namespace flowvis
} // namespace megamol
