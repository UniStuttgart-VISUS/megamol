/*
 * OpenGL_GLFW_RAPI.cpp
 *
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "OpenGL_GLFW_RAPI.hpp"

#include <array>
#include <chrono>
#include <vector>

#include "glad.h"
#ifdef _WIN32
#include <Windows.h>
#undef min
#undef max
#include "glad_wgl.h"
#else
#include "glad_glx.h"
#endif

#include "GLFW/glfw3.h"
#ifdef _WIN32
#    ifndef USE_EGL
#        define GLFW_EXPOSE_NATIVE_WIN32
#        include "GLFW/glfw3native.h"
#    endif
#endif

#include "UILayersCollection.hpp"
#include "vislib/graphics/FpsCounter.h"
#include "vislib/sys/Log.h"

#include <functional>

namespace megamol {
namespace render_api {

namespace {
// the idea is that we only borrow, but _do not_ manipulate shared data somebody else gave us
struct SharedData {
    GLFWwindow* borrowed_glfwContextWindowPtr{nullptr}; //
    // The SharedData idea is a bit broken here: on one hand, each window needs its own handle and GL context, on the
    // other hand a GL context can share its resources with others when its glfwWindow* handle gets passed to creation
    // of another Window/OpenGL context. So this handle is private, but can be used by other RAPI instances too.
};

void initSharedContext(SharedData& context) {
    context.borrowed_glfwContextWindowPtr = nullptr; // stays null since nobody shared his GL context with us
}

// indirection to not spam header file with GLFW inclucde
#define that static_cast<OpenGL_GLFW_RAPI*>(::glfwGetWindowUserPointer(wnd))
void outer_glfw_onKey_func(GLFWwindow* wnd, int k, int s, int a, int m) { that->glfw_onKey_func(k, s, a, m); }
void outer_glfw_onChar_func(GLFWwindow* wnd, unsigned int charcode) { that->glfw_onChar_func(charcode); }
void outer_glfw_onMouseMove_func(GLFWwindow* wnd, double x, double y) { that->glfw_onMouseMove_func(x, y); }
void outer_glfw_onMouseButton_func(GLFWwindow* wnd, int b, int a, int m) { that->glfw_onMouseButton_func(b, a, m); }
void outer_glfw_onMouseWheel_func(GLFWwindow* wnd, double x, double y) { that->glfw_onMouseWheel_func(x, y); }

} // namespace

// helpers to simplify data access
#define m_data (*m_pimpl)
#define m_sharedData ((m_data.sharedDataPtr) ? (*m_data.sharedDataPtr) : (m_data.sharedData))
#define m_glfwWindowPtr (m_data.glfwContextWindowPtr)

struct OpenGL_GLFW_RAPI::PimplData {
    SharedData sharedData;              // personal data we share with other RAPIs
    SharedData* sharedDataPtr{nullptr}; // if we get shared data from another OGL RAPI object, we access it using this
                                        // ptr and leave our own shared data un-initialized

    GLFWwindow* glfwContextWindowPtr{nullptr}; // _my own_ gl context!
    OpenGL_GLFW_RAPI::Config initialConfig;    // keep copy of user-provided config

    std::string fullWindowTitle;

    int currentWidth = 0, currentHeight = 0;

    UILayersCollection uiLayers;
    std::shared_ptr<AbstractUILayer> mouseCapture{nullptr};

    // TODO: move into 'FrameStatisticsCalculator'
    vislib::graphics::FpsCounter fpsCntr;
    float fps = 0.0f;
    std::array<float, 20> fpsList = {0.0f};
    bool showFpsInTitle = true;
    std::chrono::system_clock::time_point fpsSyncTime;
    GLuint fragmentQuery;
    GLuint primsQuery;
    bool showFragmentsInTitle;
    bool showPrimsInTitle;
    // glGenQueries(1, &m_data.fragmentQuery);
    // glGenQueries(1, &m_data.primsQuery);
    // m_data.fpsSyncTime = std::chrono::system_clock::now();
};


OpenGL_GLFW_RAPI::~OpenGL_GLFW_RAPI() {
    if (m_pimpl) this->closeAPI(); // cleans up pimpl
}

bool OpenGL_GLFW_RAPI::initAPI(void* configPtr) {
    if (configPtr == nullptr) return false;

    return initAPI(*static_cast<Config*>(configPtr));
}

bool OpenGL_GLFW_RAPI::initAPI(const Config& config) {
    if (m_pimpl) return false; // this API object is already initialized

    // TODO: check config for sanity?
    // if(!sane(config))
    //	return false

    m_pimpl = std::unique_ptr<PimplData, std::function<void(PimplData*)>>(new PimplData, [](PimplData* ptr) { delete ptr; });
    m_data.initialConfig = config;
    // from here on, access pimpl data using "m_data.member", as in m_pimpl->member minus the  void-ptr casting

    // init (shared) context data for this object or use provided
    if (config.sharedContextPtr) {
        m_data.sharedDataPtr = reinterpret_cast<SharedData*>(config.sharedContextPtr);
    } else {
        initSharedContext(m_data.sharedData);
    }
    // from here on, use m_sharedData to access reference to SharedData for RAPI objects; the owner will clean it up
    // correctly
    if (m_data.sharedDataPtr) {
		// glfw already initialized by other render api
    } else {
        const bool success_glfw = glfwInit();
        if (!success_glfw)
			return false; // glfw had error on init; abort
    }

    // init glfw window and OpenGL Context
    ::glfwWindowHint(GLFW_ALPHA_BITS, 8);
    ::glfwWindowHint(GLFW_DECORATED,
        (config.windowPlacement.fullScreen) ? (GL_FALSE) : (config.windowPlacement.noDec ? GL_FALSE : GL_TRUE));
    ::glfwWindowHint(GLFW_VISIBLE, GL_FALSE); // initially invisible

    int monCnt = 0;
    GLFWmonitor** monitors = ::glfwGetMonitors(&monCnt); // primary monitor is first in list
    if (!monitors) return false;                         // no monitor found; abort

    // in fullscreen, use last available monitor as to not block primary monitor, where the user may have important
    // stuff he wants to look at
    int monitorNr =
        (config.windowPlacement.fullScreen)
            ? std::max<int>(0, std::min<int>(monCnt - 1,
                                   config.windowPlacement.mon)) // if fullscreen, use last or user-provided monitor
            : (0);                                              // if windowed, use primary monitor
    GLFWmonitor* selectedMonitor = monitors[monitorNr];
    if (!selectedMonitor) return false; // selected monitor not valid for some reason; abort

    const GLFWvidmode* mode = ::glfwGetVideoMode(selectedMonitor);
    if (!mode) return false; // error while receiving monitor mode; abort

    // window size for windowed mode
    if (!config.windowPlacement.fullScreen) {
        if (config.windowPlacement.size && (config.windowPlacement.w > 0) && (config.windowPlacement.h > 0)) {
            m_data.currentWidth = config.windowPlacement.w;
            m_data.currentHeight = config.windowPlacement.h;
        } else {
            vislib::sys::Log::DefaultLog.WriteWarn("No useful window size given. Making one up");
            // no useful window size given, derive one from monitor resolution
            m_data.currentWidth = mode->width * 3 / 4;
            m_data.currentHeight = mode->height * 3 / 4;
        }
    }

    // options for fullscreen mode
    if (config.windowPlacement.fullScreen) {
        if (config.windowPlacement.pos)
            vislib::sys::Log::DefaultLog.WriteWarn("Ignoring window placement position when requesting fullscreen.");

        if (config.windowPlacement.size &&
            ((config.windowPlacement.w != mode->width) || (config.windowPlacement.h != mode->height)))
            vislib::sys::Log::DefaultLog.WriteWarn("Changing screen resolution is currently not supported.");

        if (config.windowPlacement.noDec)
            vislib::sys::Log::DefaultLog.WriteWarn("Ignoring no-decorations setting when requesting fullscreen.");

        /* note we do not use a real fullscrene mode, since then we would have focus-iconify problems */
        m_data.currentWidth = mode->width;
        m_data.currentHeight = mode->height;

        ::glfwWindowHint(GLFW_RED_BITS, mode->redBits);
        ::glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
        ::glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
        ::glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
        // this only works since we are NOT setting a monitor
        ::glfwWindowHint(GLFW_FLOATING, GL_TRUE); // floating above other windows / top most

        // will place 'fullscreen' window at origin of monitor
        int mon_x, mon_y;
        ::glfwGetMonitorPos(selectedMonitor, &mon_x, &mon_y);
        m_data.initialConfig.windowPlacement.x = mon_x;
        m_data.initialConfig.windowPlacement.y = mon_y;
    }

    // TODO: OpenGL context hints? version? core profile?
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    m_glfwWindowPtr = ::glfwCreateWindow(m_data.currentWidth, m_data.currentHeight,
        m_data.initialConfig.windowTitlePrefix.c_str(), nullptr, m_sharedData.borrowed_glfwContextWindowPtr);

    if (!m_glfwWindowPtr) {
        vislib::sys::Log::DefaultLog.WriteInfo("OpenGL_GLFW_RAPI: Failed to create GLFW window.");
        return false;
    }
    vislib::sys::Log::DefaultLog.WriteInfo(
        "OpenGL_GLFW_RAPI: Create window with size w: %d, h: %d\n", m_data.currentWidth, m_data.currentHeight);

    ::glfwMakeContextCurrent(m_glfwWindowPtr);

    if (config.windowPlacement.pos ||
        config.windowPlacement
            .fullScreen) // note the m_data window position got overwritten with monitor position for fullscreen mode
        ::glfwSetWindowPos(
            m_glfwWindowPtr, m_data.initialConfig.windowPlacement.x, m_data.initialConfig.windowPlacement.y);

	// TODO: when do we need this?
    // if (config.windowPlacement.fullScreen ||
    //     config.windowPlacement.noDec) {
    //     ::glfwSetInputMode(m_glfwWindowPtr, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    // }

	gladLoadGL();
#ifdef _WIN32
	gladLoadWGL(wglGetCurrentDC());
#else
    Display *display = XOpenDisplay(NULL);
	gladLoadGLX(display, DefaultScreen(display));
    XCloseDisplay(display);
#endif

    ::glfwSetWindowUserPointer(m_glfwWindowPtr, this); // this is ok, as long as no one derives from this RAPI
    ::glfwSetKeyCallback(m_glfwWindowPtr, &outer_glfw_onKey_func);
    ::glfwSetMouseButtonCallback(m_glfwWindowPtr, &outer_glfw_onMouseButton_func);
    ::glfwSetCursorPosCallback(m_glfwWindowPtr, &outer_glfw_onMouseMove_func);
    ::glfwSetScrollCallback(m_glfwWindowPtr, &outer_glfw_onMouseWheel_func);
    ::glfwSetCharCallback(m_glfwWindowPtr, &outer_glfw_onChar_func);

	// TODO: implement OpenGL Debug
    //if (config.enableKHRDebug)
	//    megamol::core::utility::KHR::startDebug();

	if (config.enableVsync)
		::glfwSwapInterval(0);

    ::glfwShowWindow(m_glfwWindowPtr);
    ::glfwMakeContextCurrent(nullptr);
    return true;
}

void OpenGL_GLFW_RAPI::closeAPI() {
    if (!m_pimpl) // this API object is not initialized
        return;

	const bool close_glfw = (m_data.sharedDataPtr == nullptr);

	::glfwMakeContextCurrent(m_glfwWindowPtr);

	// GL context and destruction of all other things happens in destructors of pimpl data members
	if (m_data.glfwContextWindowPtr)
		::glfwDestroyWindow(m_data.glfwContextWindowPtr);
	m_data.sharedDataPtr = nullptr;
    m_data.glfwContextWindowPtr = nullptr;
    m_data.mouseCapture = nullptr;
    this->m_pimpl.release();

	::glfwMakeContextCurrent(nullptr);

	if (close_glfw) {
        glfwTerminate();
	}
}

void OpenGL_GLFW_RAPI::preViewRender() {
    if (m_glfwWindowPtr == nullptr) return;

    // poll events for all GLFW windows. this also issues the callbacks. note at this point there is no GL context
    // active.
    ::glfwPollEvents();
    // from GLFW Docs:
    // Do not assume that callbacks will only be called through glfwPollEvents().
    // While it is necessary to process events in the event queue,
    // some window systems will send some events directly to the application,
    // which in turn causes callbacks to be called outside of regular event processing.

	// TODO: use GLFW callback
    if (::glfwWindowShouldClose(m_glfwWindowPtr))
        this->setShutdown(true); // cleanup of this RAPI and dependent GL stuff is triggered via this shutdown hint

    ::glfwMakeContextCurrent(m_glfwWindowPtr);

	// TODO: use GLFW callback
    if (checkWindowResize())
		on_resize(m_data.currentWidth, m_data.currentHeight);

    // start frame timer

    // rendering via MegaMol View is called after this function finishes
    // in the end this calls something like
    //::mmcRenderView(hView, &renderContext);
}

void OpenGL_GLFW_RAPI::postViewRender() {
    if (m_glfwWindowPtr == nullptr) return;

    // end frame timer
    // update window name

    m_data.uiLayers.OnDraw();

    ::glfwSwapBuffers(m_glfwWindowPtr);

    // TODO: in gl::Window::Update() this only got called every second or so. was there an important reason to do so?
#ifdef _WIN32
    // TODO fix this for EGL + Win
    if (m_data.initialConfig.windowPlacement.topMost) {
        vislib::sys::Log::DefaultLog.WriteInfo("Periodic reordering of windows.");
        SetWindowPos(glfwGetWin32Window(m_glfwWindowPtr), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE);
    }
#endif

    ::glfwMakeContextCurrent(nullptr);
}

const void* OpenGL_GLFW_RAPI::getAPISharedDataPtr() const { return &m_sharedData; }

void OpenGL_GLFW_RAPI::updateWindowTitle() {}

void OpenGL_GLFW_RAPI::AddUILayer(std::shared_ptr<AbstractUILayer> uiLayer) {
	m_data.uiLayers.AddUILayer(uiLayer);
}

void OpenGL_GLFW_RAPI::RemoveUILayer(std::shared_ptr<AbstractUILayer> uiLayer) {
    m_data.uiLayers.RemoveUILayer(uiLayer);
}

void OpenGL_GLFW_RAPI::glfw_onKey_func(int k, int s, int a, int m) {
    //::glfwMakeContextCurrent(m_glfwWindowPtr);

    render_api::Key key = static_cast<render_api::Key>(k);
    render_api::KeyAction action(render_api::KeyAction::RELEASE);
    switch (a) {
    case GLFW_PRESS:
        action = render_api::KeyAction::PRESS;
        break;
    case GLFW_REPEAT:
        action = render_api::KeyAction::REPEAT;
        break;
    case GLFW_RELEASE:
        action = render_api::KeyAction::RELEASE;
        break;
    }

    render_api::Modifiers mods;
    if ((m & GLFW_MOD_SHIFT) == GLFW_MOD_SHIFT) mods |= render_api::Modifier::SHIFT;
    if ((m & GLFW_MOD_CONTROL) == GLFW_MOD_CONTROL) mods |= render_api::Modifier::CTRL;
    if ((m & GLFW_MOD_ALT) == GLFW_MOD_ALT) mods |= render_api::Modifier::ALT;

    //m_data.uiLayers.OnKey(key, action, mods);
    this->ui_events.onKey_list.emplace_back(std::make_tuple(key, action, mods));
}

void OpenGL_GLFW_RAPI::glfw_onChar_func(unsigned int charcode) {
    //::glfwMakeContextCurrent(m_glfwWindowPtr);
    //m_data.uiLayers.OnChar(charcode);
    this->ui_events.onChar_list.emplace_back(charcode);
}

void OpenGL_GLFW_RAPI::glfw_onMouseMove_func(double x, double y) {
    //::glfwMakeContextCurrent(m_glfwWindowPtr);
    //if (m_data.mouseCapture) {
    //    m_data.mouseCapture->OnMouseMove(x, y);
    //} else {
    //    m_data.uiLayers.OnMouseMove(x, y);
    //}
    this->ui_events.onMouseMove_list.emplace_back(std::make_tuple(x, y));
}

void OpenGL_GLFW_RAPI::glfw_onMouseButton_func(int b, int a, int m) {
    //::glfwMakeContextCurrent(m_glfwWindowPtr);
    render_api::MouseButton btn = static_cast<render_api::MouseButton>(b);
    render_api::MouseButtonAction action =
        (a == GLFW_PRESS) ? render_api::MouseButtonAction::PRESS : render_api::MouseButtonAction::RELEASE;

    render_api::Modifiers mods;
    if ((m & GLFW_MOD_SHIFT) == GLFW_MOD_SHIFT) mods |= render_api::Modifier::SHIFT;
    if ((m & GLFW_MOD_CONTROL) == GLFW_MOD_CONTROL) mods |= render_api::Modifier::CTRL;
    if ((m & GLFW_MOD_ALT) == GLFW_MOD_ALT) mods |= render_api::Modifier::ALT;

	this->ui_events.onMouseButton_list.emplace_back(std::make_tuple(btn, action, mods));

    //if (m_data.mouseCapture) {
    //    m_data.mouseCapture->OnMouseButton(btn, action, mods);
    //} else {
    //    if (m_data.uiLayers.OnMouseButton(btn, action, mods))
    //        if (action == render_api::MouseButtonAction::PRESS)
    //            m_data.mouseCapture = m_data.uiLayers.lastEventCaptureUILayer();
    //}

    //if (m_data.mouseCapture) {
    //    bool anyPressed = false;
    //    for (int mbi = GLFW_MOUSE_BUTTON_1; mbi <= GLFW_MOUSE_BUTTON_LAST; ++mbi) {
    //        if (::glfwGetMouseButton(m_glfwWindowPtr, mbi) == GLFW_PRESS) {
    //            anyPressed = true;
    //            break;
    //        }
    //    }
    //    if (!anyPressed) {
    //        m_data.mouseCapture.reset();
    //        double x, y;
    //        ::glfwGetCursorPos(m_glfwWindowPtr, &x, &y);
    //        glfw_onMouseMove_func(x, y); // to inform all of the new location
    //    }
    //}
}

void OpenGL_GLFW_RAPI::glfw_onMouseWheel_func(double x, double y) {
    //::glfwMakeContextCurrent(m_glfwWindowPtr);

	this->ui_events.onMouseWheel_list.emplace_back(std::make_tuple(x, y));
    //if (m_data.mouseCapture) {
    //    m_data.mouseCapture->OnMouseScroll(x, y);
    //} else {
    //    m_data.uiLayers.OnMouseScroll(x, y);
    //}
}

bool OpenGL_GLFW_RAPI::checkWindowResize() {
    int frame_width, frame_height;
    ::glfwGetFramebufferSize(m_glfwWindowPtr, &frame_width, &frame_height);

    if ((frame_width != m_data.currentWidth) || (frame_height != m_data.currentHeight)) {
        m_data.currentWidth = frame_width;
        m_data.currentHeight = frame_height;
        this->ui_events.is_window_resized = true;
    }
    this->ui_events.is_window_resized = false;

	return this->ui_events.is_window_resized;
}
void OpenGL_GLFW_RAPI::on_resize(int w, int h) {
    //::glfwMakeContextCurrent(m_glfwWindowPtr);
    if ((w > 0) && (h > 0)) {
        // TODO: whose responsibility? put into callback!
		// TODO: talk to karsten about this, View3D should not use GL calls. this means the RenderAPI needs to provide glViewport / FBO / resize info
        //::glViewport(0, 0, w, h);
        //::mmcResizeView(hView, w, h);
        vislib::sys::Log::DefaultLog.WriteInfo("OpenGL_GLFW_RAPI: Resize window (w: %d, h: %d)\n", w, h);

        this->ui_events.resize_list.emplace_back(w, h);
        //m_data.uiLayers.OnResize(w, h);
    }
}

OpenGL_GLFW_RAPI::UIEvents& OpenGL_GLFW_RAPI::getUIEvents() {
	return this->ui_events;
}

// TODO: how to force/allow subclasses to interop with other APIs?
// TODO: API-specific options in subclasses?

} // namespace render_api
} // namespace megamol
