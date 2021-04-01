/*
 * Screenshot_Service.cpp
 *
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "Screenshot_Service.hpp"

 // to grab GL front buffer
#include <glad/glad.h>
#include "IOpenGL_Context.h"
#include "GUI_Resource.h"

#include "mmcore/MegaMolGraph.h"

// to write png files
#include "png.h"
#include "mmcore/utility/graphics/ScreenShotComments.h"
#include "vislib/sys/FastFile.h"


// local logging wrapper for your convenience
#include "mmcore/utility/log/Log.h"
static void log(const char* text) {
    const std::string msg = "Screenshot_Service: " + std::string(text);
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}
static void log(std::string text) { log(text.c_str()); }

// need this to pass GL context to screenshot source. this a hack and needs to be properly designed.
static megamol::frontend_resources::IOpenGL_Context* gl_context_ptr = nullptr;
static megamol::core::MegaMolGraph* megamolgraph_ptr = nullptr;
static megamol::frontend_resources::GUIResource* guiresources_ptr = nullptr;

static void PNGAPI pngErrorFunc(png_structp pngPtr, png_const_charp msg) {
    log("PNG Error: " + std::string(msg));
}

static void PNGAPI pngWarnFunc(png_structp pngPtr, png_const_charp msg) {
    log("PNG Warning: " + std::string(msg));
}

static void PNGAPI pngWriteFileFunc(png_structp pngPtr, png_bytep buf, png_size_t size) {
    vislib::sys::File* f = static_cast<vislib::sys::File*>(png_get_io_ptr(pngPtr));
    f->Write(buf, size);
}

static void PNGAPI pngFlushFileFunc(png_structp pngPtr) {
    vislib::sys::File* f = static_cast<vislib::sys::File*>(png_get_io_ptr(pngPtr));
    f->Flush();
}

static bool write_png_to_file(megamol::frontend_resources::ImageData const& image, std::string const& filename) {
    vislib::sys::FastFile file;
    try {
        // open final image file
        if (!file.Open(filename.c_str(), vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::CREATE_OVERWRITE))
        {
            log("Cannot open output file" + filename);
            return false;
        }
    } catch (...) {
        log("Error/Exception opening output file" + filename);
        return false;
    }

    png_structp pngPtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, &pngErrorFunc, &pngWarnFunc);
    if (!pngPtr) {
        log("Cannot create png structure");
        return false;
    }

    png_infop pngInfoPtr = png_create_info_struct(pngPtr);
    if (!pngInfoPtr) {
        log("Cannot create png info");
        return false;
    }
    
    png_set_write_fn(pngPtr, static_cast<void*>(&file), &pngWriteFileFunc, &pngFlushFileFunc);
    
    png_set_compression_level(pngPtr, 0);

    // todo: camera settings are not stored without magic knowledge about the view
    std::string project = megamolgraph_ptr->Convenience().SerializeGraph();
    project.append(guiresources_ptr->request_gui_state(true));
    megamol::core::utility::graphics::ScreenShotComments ssc(project);
    png_set_text(pngPtr, pngInfoPtr, ssc.GetComments().data(), ssc.GetComments().size());

    png_set_IHDR(pngPtr, pngInfoPtr, image.width, image.height, 8,
        PNG_COLOR_TYPE_RGB_ALPHA /* PNG_COLOR_TYPE_RGB */, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_set_rows(pngPtr, pngInfoPtr, const_cast<png_byte**>(reinterpret_cast<png_byte* const*>(image.flipped_rows.data())));

    png_write_png(pngPtr, pngInfoPtr, PNG_TRANSFORM_IDENTITY, NULL);

    png_destroy_write_struct(&pngPtr, &pngInfoPtr);

    return true;
}

void megamol::frontend_resources::GLScreenshotSource::set_read_buffer(ReadBuffer buffer) {
    m_read_buffer = buffer;
    GLenum read_buffer;

    switch (buffer) {
    default:
        [[fallthrough]];
    case ReadBuffer::FRONT:
            read_buffer = GL_FRONT;
        break;
    case ReadBuffer::BACK:
            read_buffer = GL_BACK;
        break;
    case ReadBuffer::COLOR_ATT0:
            read_buffer = GL_COLOR_ATTACHMENT0;
        break;
    case ReadBuffer::COLOR_ATT1:
            read_buffer = GL_COLOR_ATTACHMENT0+1;
        break;
    case ReadBuffer::COLOR_ATT2:
            read_buffer = GL_COLOR_ATTACHMENT0+2;
        break;
    case ReadBuffer::COLOR_ATT3:
            read_buffer = GL_COLOR_ATTACHMENT0+3;
        break;
    }
}

megamol::frontend_resources::ImageData megamol::frontend_resources::GLScreenshotSource::take_screenshot() const {
    if (gl_context_ptr)
        gl_context_ptr->activate();

    // TODO: in FBO-based rendering the FBO object carries its size and we dont need to look it up
    // simpler and more correct approach would be to observe Framebuffer_Events resource
    // but this is our naive implementation for now
    GLint viewport_dims[4] = {0};
    glGetIntegerv(GL_VIEWPORT, viewport_dims);
    GLint fbWidth = viewport_dims[2];
    GLint fbHeight = viewport_dims[3];

    ImageData result;
    result.resize(static_cast<size_t>(fbWidth), static_cast<size_t>(fbHeight));

    glReadBuffer(m_read_buffer);
    glReadPixels(0, 0, fbWidth, fbHeight, GL_RGBA, GL_UNSIGNED_BYTE, result.image.data());

    for (auto& pixel : result.image)
        pixel.a = 255;

    if (gl_context_ptr)
        gl_context_ptr->close();

    return std::move(result);
}

bool megamol::frontend_resources::ImageDataToPNGWriter::write_image(ImageData image, std::string const& filename) const {
    return write_png_to_file(std::move(image), filename);
}

namespace megamol {
namespace frontend {

Screenshot_Service::Screenshot_Service() {}

Screenshot_Service::~Screenshot_Service() {}

bool Screenshot_Service::init(void* configPtr) {
    if (configPtr == nullptr) return false;

    return init(*static_cast<Config*>(configPtr));
}

bool Screenshot_Service::init(const Config& config) {

    m_requestedResourcesNames =
    {
        "IOpenGL_Context",
        "MegaMolGraph",
        "GUIResource"
    };

    this->m_frontbufferToPNG_trigger = [&](std::string const& filename) -> bool
    {
        log("write screenshot to " + filename);
        return m_toFileWriter_resource.write_screenshot(m_frontbufferSource_resource, filename);
    };

    log("initialized successfully");
    return true;
}

void Screenshot_Service::close() {
    // close libraries or APIs you manage
    // wrap up resources your service provides, but don not depend on outside resources to be available here
    // after this, at some point only the destructor of your service gets called
}

std::vector<FrontendResource>& Screenshot_Service::getProvidedResources() {
     this->m_providedResourceReferences =
    {
        {"GLScreenshotSource", m_frontbufferSource_resource},
        {"ImageDataToPNGWriter", m_toFileWriter_resource},
        {"GLFrontbufferToPNG_ScreenshotTrigger", m_frontbufferToPNG_trigger}
    };

    return m_providedResourceReferences;
}

const std::vector<std::string> Screenshot_Service::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

void Screenshot_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    gl_context_ptr = const_cast<megamol::frontend_resources::IOpenGL_Context*>(&resources[0].getResource<megamol::frontend_resources::IOpenGL_Context>());
    megamolgraph_ptr = const_cast<megamol::core::MegaMolGraph*>(&resources[1].getResource<megamol::core::MegaMolGraph>());
    guiresources_ptr = const_cast<megamol::frontend_resources::GUIResource*>(
        &resources[2].getResource<megamol::frontend_resources::GUIResource>());
}

void Screenshot_Service::updateProvidedResources() {
}

void Screenshot_Service::digestChangedRequestedResources() {
    bool need_to_shutdown = false;
    if (need_to_shutdown)
        this->setShutdown();
}

void Screenshot_Service::resetProvidedResources() {
    // this gets called at the end of the main loop iteration
    // since the current resources state should have been handled in this frame already
    // you may clean up resources whose state is not needed for the next iteration
    // e.g. m_keyboardEvents.clear();
    // network_traffic_buffer.reset_to_empty();
}

void Screenshot_Service::preGraphRender() {
    // this gets called right before the graph is told to render something
    // e.g. you can start a start frame timer here

    // rendering via MegaMol View is called after this function finishes
    // in the end this calls the equivalent of ::mmcRenderView(hView, &renderContext)
    // which leads to view.Render()
}

void Screenshot_Service::postGraphRender() {
    // the graph finished rendering and you may more stuff here
    // e.g. end frame timer
    // update window name
    // swap buffers, glClear
}


} // namespace frontend
} // namespace megamol
