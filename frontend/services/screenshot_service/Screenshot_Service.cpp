/*
 * Screenshot_Service.cpp
 *
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "Screenshot_Service.hpp"

#include "GUIState.h"
#include "OpenGL_Context.h"

#include "ImageWrapper.h"
#include "ImageWrapper_to_ByteArray.hpp"

#include "mmcore/MegaMolGraph.h"

// to write png files
#include "mmcore/utility/graphics/ScreenShotComments.h"
#include "png.h"
#include "vislib/sys/FastFile.h"
#include "zlib.h"

#include "mmcore/utility/log/Log.h"

#include "GUIRegisterWindow.h"

static std::shared_ptr<bool> service_open_popup = std::make_shared<bool>(false);
static const std::string service_name = "Screenshot_Service: ";
static const std::string privacy_note(
    "--- PRIVACY NOTE ---\n"
    "Please note that the complete MegaMol project is stored in the header of the screenshot image file. \n"
    "Before giving away the screenshot, clear privacy relevant information in the project file before taking a "
    "screenshot (e.g. user name in file paths). \n"
    ">>> In the file [megamol_config.lua] set mmSetCliOption(\"privacynote\", \"off\") to permanently turn off privacy "
    "notifications for screenshots.");

static void log(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}

static void log_error(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
}

static void log_warning(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteWarn(msg.c_str());
}

// need this to pass GL context to screenshot source. this a hack and needs to be properly designed.
static megamol::core::MegaMolGraph* megamolgraph_ptr = nullptr;
static megamol::frontend_resources::GUIState* guistate_resources_ptr = nullptr;
static bool screenshot_show_privacy_note = true;

unsigned char megamol::frontend::Screenshot_Service::default_alpha_value = 255;

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

static bool write_png_to_file(
    megamol::frontend_resources::ScreenshotImageData const& image, std::filesystem::path const& filename) {
    vislib::sys::FastFile file;
    try {
        // open final image file
        if (!file.Open(filename.native().c_str(), vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE,
                vislib::sys::File::CREATE_OVERWRITE)) {
            log("Cannot open output file" + filename.generic_u8string());
            return false;
        }
    } catch (...) {
        log("Error/Exception opening output file" + filename.generic_u8string());
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

    png_set_compression_level(pngPtr, Z_BEST_SPEED);

    // todo: camera settings are not stored without magic knowledge about the view
    std::string project = megamolgraph_ptr->Convenience().SerializeGraph();
    if (guistate_resources_ptr) {
        project.append(guistate_resources_ptr->request_gui_state(true));
    }
    megamol::core::utility::graphics::ScreenShotComments ssc(project);
    png_set_text(pngPtr, pngInfoPtr, ssc.GetComments().data(), ssc.GetComments().size());

    png_set_IHDR(pngPtr, pngInfoPtr, image.width, image.height, 8, PNG_COLOR_TYPE_RGB_ALPHA /* PNG_COLOR_TYPE_RGB */,
        PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_set_rows(
        pngPtr, pngInfoPtr, const_cast<png_byte**>(reinterpret_cast<png_byte* const*>(image.flipped_rows.data())));

    png_write_png(pngPtr, pngInfoPtr, PNG_TRANSFORM_IDENTITY, NULL);

    png_destroy_write_struct(&pngPtr, &pngInfoPtr);

    file.Close();

    if (screenshot_show_privacy_note) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("Screenshot: %s", privacy_note.c_str());
        if (service_open_popup != nullptr)
            *service_open_popup = true;
    }
    return true;
}

megamol::frontend_resources::ImageWrapperScreenshotSource::ImageWrapperScreenshotSource(ImageWrapper const& image)
        : m_image{&const_cast<ImageWrapper&>(image)} {}

megamol::frontend_resources::ScreenshotImageData const&
megamol::frontend_resources::ImageWrapperScreenshotSource::take_screenshot() const {
    static ScreenshotImageData screenshot_image;

    // keep allocated vector memory around
    // note that this initially holds a nullptr texture - bad!

    static frontend_resources::byte_texture image_bytes({});

    // fill bytes with image data
    image_bytes = *m_image;
    if (m_image->channels != ImageWrapper::DataChannels::RGBA8 &&
        m_image->channels != ImageWrapper::DataChannels::RGB8) {
        throw std::runtime_error("[Screenshot_Service] Only image with RGBA8 or RGA8 channels supported for now...");
    }

    auto& byte_vector = image_bytes.as_byte_vector();
    screenshot_image.resize(m_image->size.width, m_image->size.height);

    if (byte_vector.size() !=
        (screenshot_image.image.size() * ((m_image->channels == ImageWrapper::DataChannels::RGBA8) ? (4) : (3)))) {
        throw std::runtime_error("[Screenshot_Service] Image is not correctly initialized...");
    }

    for (size_t i = 0, j = 0; i < byte_vector.size();) {
        auto r = [&]() { return byte_vector[i++]; };
        auto g = [&]() { return byte_vector[i++]; };
        auto b = [&]() { return byte_vector[i++]; };
        auto a = [&]() {
            return (m_image->channels == ImageWrapper::DataChannels::RGBA8)
                       ? byte_vector[i++]
                       : megamol::frontend::Screenshot_Service::default_alpha_value;
        }; // alpha either from image or 1.0
        ScreenshotImageData::Pixel pixel = {r(), g(), b(), a()};
        screenshot_image.image[j++] = pixel;
    }

    return screenshot_image;
}

bool megamol::frontend_resources::ScreenshotImageDataToPNGWriter::write_image(
    ScreenshotImageData const& image, std::filesystem::path const& filename) const {
    return write_png_to_file(image, filename);
}

namespace megamol {
namespace frontend {

Screenshot_Service::Screenshot_Service() {}

Screenshot_Service::~Screenshot_Service() {
    service_open_popup.reset();
}

bool Screenshot_Service::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    return init(*static_cast<Config*>(configPtr));
}

bool Screenshot_Service::init(const Config& config) {

    m_requestedResourcesNames = {"optional<OpenGL_Context>", // TODO: for GLScreenshoSource. how to kill?
        "MegaMolGraph", "optional<GUIState>", "RuntimeConfig", "optional<GUIRegisterWindow>"};

    this->m_frontbufferToPNG_trigger = [&](std::filesystem::path const& filename) -> bool {
        log("write screenshot to " + filename.generic_u8string());
        return m_toFileWriter_resource.write_screenshot(m_frontbufferSource_resource, filename);
    };

    screenshot_show_privacy_note = config.show_privacy_note;

    this->m_imagewrapperToPNG_trigger = [&](megamol::frontend_resources::ImageWrapper const& image,
                                            std::filesystem::path const& filename) -> bool {
        log("write screenshot to " + filename.generic_u8string());
        return m_toFileWriter_resource.write_screenshot(
            megamol::frontend_resources::ImageWrapperScreenshotSource(image), filename);
    };

    log("initialized successfully");
    return true;
}

void Screenshot_Service::close() {}

std::vector<FrontendResource>& Screenshot_Service::getProvidedResources() {
    this->m_providedResourceReferences = {{"GLScreenshotSource", m_frontbufferSource_resource},
        {"ImageDataToPNGWriter", m_toFileWriter_resource},
        {"GLFrontbufferToPNG_ScreenshotTrigger", m_frontbufferToPNG_trigger},
        {"ImageWrapperToPNG_ScreenshotTrigger", m_imagewrapperToPNG_trigger}};


    return m_providedResourceReferences;
}

const std::vector<std::string> Screenshot_Service::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

void Screenshot_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    megamolgraph_ptr =
        const_cast<megamol::core::MegaMolGraph*>(&resources[1].getResource<megamol::core::MegaMolGraph>());

    auto maybe_gui_state = resources[2].getOptionalResource<megamol::frontend_resources::GUIState>();
    if (maybe_gui_state.has_value()) {
        guistate_resources_ptr = const_cast<megamol::frontend_resources::GUIState*>(&maybe_gui_state.value().get());
    }

    auto maybe_gui_window_request_resource =
        resources[4].getOptionalResource<megamol::frontend_resources::GUIRegisterWindow>();
    if (maybe_gui_window_request_resource.has_value()) {
        auto& gui_window_request_resource = maybe_gui_window_request_resource.value().get();
        gui_window_request_resource.register_notification(
            "Screenshot", std::weak_ptr<bool>(service_open_popup), privacy_note);
    }
}

void Screenshot_Service::updateProvidedResources() {}

void Screenshot_Service::digestChangedRequestedResources() {
    bool need_to_shutdown = false;
    if (need_to_shutdown)
        this->setShutdown();
}

void Screenshot_Service::resetProvidedResources() {}

void Screenshot_Service::preGraphRender() {}

void Screenshot_Service::postGraphRender() {}


} // namespace frontend
} // namespace megamol
