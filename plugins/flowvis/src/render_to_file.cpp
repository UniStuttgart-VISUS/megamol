/*
 * render_to_file.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"

#include "render_to_file.h"

#include "mmcore/job/TickCall.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/view/CallRenderView.h"

#include "vislib/Trace.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/Log.h"

#include "png.h"

#include <array>
#include <memory>
#include <vector>

namespace {
/**
 * My error handling function for png export
 *
 * @param pngPtr The png structure pointer
 * @param msg The error message
 */
static void PNGAPI myPngError(png_structp pngPtr, png_const_charp msg) {
    vislib::sys::Log::DefaultLog.WriteError("%s", msg);
}

/**
 * My error handling function for png export
 *
 * @param pngPtr The png structure pointer
 * @param msg The error message
 */
static void PNGAPI myPngWarn(png_structp pngPtr, png_const_charp msg) {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, "Png-Warning: %s\n", msg);
}

/**
 * My write function for png export
 *
 * @param pngPtr The png structure pointer
 * @param buf The pointer to the buffer to be written
 * @param size The number of bytes to be written
 */
static void PNGAPI myPngWrite(png_structp pngPtr, png_bytep buf, png_size_t size) {
    vislib::sys::File* f = static_cast<vislib::sys::File*>(png_get_io_ptr(pngPtr));
    f->Write(buf, size);
}

/**
 * My flush function for png export
 *
 * @param pngPtr The png structure pointer
 */
static void PNGAPI myPngFlush(png_structp pngPtr) {
    vislib::sys::File* f = static_cast<vislib::sys::File*>(png_get_io_ptr(pngPtr));
    f->Flush();
}
} // namespace

namespace megamol {
namespace flowvis {

/*
 * render_to_file::render_to_file
 */
render_to_file::render_to_file(void)
    : AbstractView()
    , viewSlot("view", "Connects to a view")
    , override_view_call(nullptr)
    , fbo_width("width", "Output width")
    , fbo_height("height", "Output height")
    , fbo_viewport("viewport", "Viewport options to set viewport of file output")
    , output_file("output_file", "Output file path")
    , trigger("trigger", "Write the rendering to file") {

    this->viewSlot.SetCompatibleCall<core::view::CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->viewSlot);

    this->fbo_width << new core::param::IntParam(1920);
    this->MakeSlotAvailable(&this->fbo_width);

    this->fbo_height << new core::param::IntParam(1080);
    this->MakeSlotAvailable(&this->fbo_height);

    this->fbo_viewport << new core::param::EnumParam(0);
    this->fbo_viewport.Param<core::param::EnumParam>()->SetTypePair(0, "Screenshot");
    this->fbo_viewport.Param<core::param::EnumParam>()->SetTypePair(1, "FBO size");
    this->MakeSlotAvailable(&this->fbo_viewport);

    this->output_file << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->output_file);

    this->trigger << new core::param::ButtonParam(core::view::Key::KEY_S, core::view::Modifier::ALT);
    this->trigger.SetUpdateCallback(&render_to_file::record);
    this->MakeSlotAvailable(&this->trigger);
}


/*
 * render_to_file::~render_to_file
 */
render_to_file::~render_to_file(void) { this->Release(); }


/*
 * render_to_file::DefaultTime
 */
float render_to_file::DefaultTime(double instTime) const {
    // This view does not do any time control
    return 0.0f;
}


/*
 * render_to_file::GetCameraSyncNumber
 */
unsigned int render_to_file::GetCameraSyncNumber(void) const {
    vislib::sys::Log::DefaultLog.WriteWarn("render_to_file::GetCameraSyncNumber unsupported");
    return 0u;
}


/*
 * render_to_file::SerialiseCamera
 */
void render_to_file::SerialiseCamera(vislib::Serialiser& serialiser) const {
    vislib::sys::Log::DefaultLog.WriteWarn("render_to_file::SerialiseCamera unsupported");
}


/*
 * render_to_file::DeserialiseCamera
 */
void render_to_file::DeserialiseCamera(vislib::Serialiser& serialiser) {
    vislib::sys::Log::DefaultLog.WriteWarn("render_to_file::DeserialiseCamera unsupported");
}


/*
 * render_to_file::Render
 */
void render_to_file::Render(const mmcRenderViewContext& context) {
    auto* view = this->viewSlot.CallAs<core::view::CallRenderView>();

    if (view != nullptr) {
        std::unique_ptr<core::view::CallRenderView> last_view_call = nullptr;

        if (this->override_view_call != nullptr) {
            last_view_call = std::make_unique<core::view::CallRenderView>(*view);
            *view = *this->override_view_call;
        } else {
            const_cast<vislib::math::Rectangle<int>&>(view->GetViewport()).Set(0, 0, this->vp_width, this->vp_height);
        }

        view->SetInstanceTime(context.InstanceTime);
        view->SetTime(static_cast<float>(context.Time));

        if (this->doHookCode()) {
            this->doBeforeRenderHook();
        }

        (*view)(core::view::CallRenderView::CALL_RENDER);

        if (this->doHookCode()) {
            this->doAfterRenderHook();
        }

        if (last_view_call != nullptr) {
            *view = *last_view_call;
        }
    }
}


/*
 * render_to_file::record
 */
bool render_to_file::record(core::param::ParamSlot&) {
    // Get parameters
    const auto fbo_viewport = this->fbo_viewport.Param<core::param::EnumParam>()->Value();

    const auto fbo_width = fbo_viewport == 0
                               ? this->width
                               : static_cast<unsigned int>(this->fbo_width.Param<core::param::IntParam>()->Value());
    const auto fbo_height = fbo_viewport == 0
                                ? this->height
                                : static_cast<unsigned int>(this->fbo_height.Param<core::param::IntParam>()->Value());

    const auto output_file = this->output_file.Param<core::param::FilePathParam>()->Value();

    // Set up fbo
    vislib::graphics::gl::FramebufferObject fbo;

    std::array<vislib::graphics::gl::FramebufferObject::ColourAttachParams, 1> cap;
    cap[0].internalFormat = GL_RGBA8;
    cap[0].format = GL_RGBA;
    cap[0].type = GL_UNSIGNED_BYTE;

    vislib::graphics::gl::FramebufferObject::DepthAttachParams dap;
    dap.format = GL_DEPTH_COMPONENT24;
    dap.state = vislib::graphics::gl::FramebufferObject::ATTACHMENT_DISABLED;

    vislib::graphics::gl::FramebufferObject::StencilAttachParams sap;
    sap.format = GL_STENCIL_INDEX;
    sap.state = vislib::graphics::gl::FramebufferObject::ATTACHMENT_DISABLED;

    fbo.Create(fbo_width, fbo_height, cap.size(), cap.data(), dap, sap);
    fbo.Disable();

    // Save state
    std::array<float, 4> clear_color;
    glGetFloatv(GL_COLOR_CLEAR_VALUE, clear_color.data());

    // Set viewport
    this->vp_width = fbo_width;
    this->vp_height = fbo_height;

    auto* view = this->viewSlot.CallAs<core::view::CallRenderView>();

    if (view != nullptr) {
        AbstractView* abstract_view = const_cast<AbstractView*>(
            dynamic_cast<const AbstractView*>(static_cast<const Module*>(view->PeekCalleeSlot()->Owner())));

        if (abstract_view != nullptr) {
            abstract_view->Resize(this->vp_width, this->vp_height);
        }
    }

    // Render
    fbo.Enable();

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));

    context.Time = this->time;
    context.InstanceTime = this->instance_time;

    this->Render(context);

    fbo.Disable();

    // Restore viewport
    this->vp_width = this->width;
    this->vp_height = this->height;

    if (view != nullptr) {
        AbstractView* abstract_view = const_cast<AbstractView*>(
            dynamic_cast<const AbstractView*>(static_cast<const Module*>(view->PeekCalleeSlot()->Owner())));

        if (abstract_view != nullptr) {
            abstract_view->Resize(this->width, this->height);
        }
    }

    // Restore state
    glClearColor(clear_color[0], clear_color[1], clear_color[2], clear_color[3]);

    // Download texture data
    static_assert(sizeof(GLubyte) == sizeof(uint8_t) && sizeof(uint8_t) == sizeof(png_byte), "Mismatching data types");

    std::vector<uint8_t> buffer(static_cast<std::size_t>(fbo_width) * fbo_height * 4 * sizeof(GLubyte));

    fbo.BindColourTexture(0);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, buffer.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    // Save to file
    vislib::sys::FastFile file;

    if (!file.Open(output_file.PeekBuffer(), vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE,
            vislib::sys::File::CREATE_OVERWRITE)) {
        vislib::sys::Log::DefaultLog.WriteError("Cannot open output file");
        return false;
    }

    auto png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, &myPngError, &myPngWarn);
    if (png_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("Cannot create png structure");
        return false;
    }

    auto png_info_ptr = png_create_info_struct(png_ptr);
    if (png_info_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("Cannot create png info");
        return false;
    }

    png_set_write_fn(png_ptr, static_cast<void*>(&file), &myPngWrite, &myPngFlush);
    png_set_IHDR(png_ptr, png_info_ptr, fbo_width, fbo_height, 8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    try {
        std::vector<uint8_t*> rows(fbo_height);

        for (std::size_t i = 0; i < rows.size(); ++i) {
            rows[rows.size() - (1 + i)] = buffer.data() + sizeof(uint8_t) * i * 4 * fbo_width;
        }

        png_set_rows(png_ptr, png_info_ptr, rows.data());
        png_write_png(png_ptr, png_info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Cannot write png file");
        return false;
    }

    if (png_ptr != nullptr) {
        if (png_info_ptr != nullptr) {
            png_destroy_write_struct(&png_ptr, &png_info_ptr);
        } else {
            png_destroy_write_struct(&png_ptr, nullptr);
        }
    }

    file.Close();

    vislib::sys::Log::DefaultLog.WriteInfo("Saved screenshot to file '%s'", output_file.PeekBuffer());
}


/*
 * render_to_file::ResetView
 */
void render_to_file::ResetView(void) {
    auto* view = this->viewSlot.CallAs<core::view::CallRenderView>();

    if (view != nullptr) (*view)(core::view::CallRenderView::CALL_RESETVIEW);
}


/*
 * render_to_file::Resize
 */
void render_to_file::Resize(unsigned int width, unsigned int height) {
    auto* view = this->viewSlot.CallAs<core::view::CallRenderView>();

    this->width = width;
    this->height = height;

    if (view != nullptr) {
        AbstractView* abstract_view = const_cast<AbstractView*>(
            dynamic_cast<const AbstractView*>(static_cast<const Module*>(view->PeekCalleeSlot()->Owner())));

        if (abstract_view != nullptr) {
            abstract_view->Resize(width, height);
        }
    }
}


/*
 * render_to_file::OnRenderView
 */
bool render_to_file::OnRenderView(core::Call& call) {
    auto* view = dynamic_cast<core::view::CallRenderView*>(&call);
    if (view == nullptr) return false;

    this->override_view_call = view;

    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));

    context.Time = this->time = view->Time();
    context.InstanceTime = this->instance_time = view->InstanceTime();

    this->Render(context);

    this->override_view_call = nullptr;

    return true;
}


/*
 * render_to_file::UpdateFreeze
 */
void render_to_file::UpdateFreeze(bool freeze) {
    auto* view = this->viewSlot.CallAs<core::view::CallRenderView>();

    if (view != nullptr)
        (*view)(freeze ? core::view::CallRenderView::CALL_FREEZE : core::view::CallRenderView::CALL_UNFREEZE);
}


bool render_to_file::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {

    bool consumed = false;

    auto* view = this->viewSlot.CallAs<core::view::CallRenderView>();
    if (view != nullptr) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;

        view->SetInputEvent(evt);

        if ((*view)(core::view::CallRenderView::FnOnKey)) consumed = true;
    }

    return consumed;
}


bool render_to_file::OnChar(unsigned int codePoint) {

    bool consumed = false;

    auto* view = this->viewSlot.CallAs<core::view::CallRenderView>();

    if (view != nullptr) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::Char;
        evt.charData.codePoint = codePoint;

        view->SetInputEvent(evt);

        if ((*view)(core::view::CallRenderView::FnOnChar)) consumed = true;
    }

    return consumed;
}


bool render_to_file::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    auto* view = this->viewSlot.CallAs<core::view::CallRenderView>();

    if (view != nullptr) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;

        view->SetInputEvent(evt);

        if ((*view)(core::view::CallRenderView::FnOnMouseButton)) return true;
    }

    return true;
}


bool render_to_file::OnMouseMove(double x, double y) {

    auto* view = this->viewSlot.CallAs<core::view::CallRenderView>();

    if (view != nullptr) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;

        view->SetInputEvent(evt);

        if ((*view)(core::view::CallRenderView::FnOnMouseMove)) return true;
    }

    return true;
}


bool render_to_file::OnMouseScroll(double dx, double dy) {

    auto* view = this->viewSlot.CallAs<core::view::CallRenderView>();

    if (view != nullptr) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;

        view->SetInputEvent(evt);

        if ((*view)(core::view::CallRenderView::FnOnMouseScroll)) return true;
    }

    return true;
}


/*
 * render_to_file::create
 */
bool render_to_file::create(void) {
    // nothing to do
    return true;
}


/*
 * render_to_file::release
 */
void render_to_file::release(void) {}


/*
 * render_to_file::unpackMouseCoordinates
 */
void render_to_file::unpackMouseCoordinates(float& x, float& y) {}

} // namespace flowvis
} // namespace megamol
