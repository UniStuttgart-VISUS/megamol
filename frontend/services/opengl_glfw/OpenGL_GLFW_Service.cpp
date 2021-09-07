/*
 * OpenGL_GLFW_Service.cpp
 *
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "OpenGL_GLFW_Service.hpp"

#include "FrameStatistics.h"

#include <array>
#include <chrono>
#include <vector>

#ifdef MEGAMOL_USE_BOOST_STACKTRACE
#define BOOST_STACKTRACE_USE_BACKTRACE
#include <boost/stacktrace.hpp>
#endif

#include "glad/glad.h"
#ifdef _WIN32
#    include <Windows.h>
#    include <DbgHelp.h>
#pragma comment(lib, "dbghelp.lib")
#    undef min
#    undef max
#    include "glad/glad_wgl.h"
#else
#    include "glad/glad_glx.h"
#endif

#include <GLFW/glfw3.h>
#ifdef _WIN32
#    ifndef USE_EGL
#        define GLFW_EXPOSE_NATIVE_WGL
#        define GLFW_EXPOSE_NATIVE_WIN32
#        include <GLFW/glfw3native.h>
#    endif
#endif

#include <functional>
#include <iostream>

#include "mmcore/utility/log/Log.h"

static const std::string service_name = "OpenGL_GLFW_Service: ";
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

static void glfw_error_callback(int error, const char* description)
{
    log_error("[GLFW Error] " + std::to_string(error) + ": " + description);
}


// See: https://github.com/glfw/glfw/issues/1630
static int fixGlfwKeyboardMods(int mods, int key, int action) {
    if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) {
        return (action == GLFW_RELEASE) ? mods & (~GLFW_MOD_SHIFT) : mods | GLFW_MOD_SHIFT;
    }
    if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) {
        return (action == GLFW_RELEASE) ? mods & (~GLFW_MOD_CONTROL) : mods | GLFW_MOD_CONTROL;
    }
    if (key == GLFW_KEY_LEFT_ALT || key == GLFW_KEY_RIGHT_ALT) {
        return (action == GLFW_RELEASE) ? mods & (~GLFW_MOD_ALT) : mods | GLFW_MOD_ALT;
    }
    if (key == GLFW_KEY_LEFT_SUPER || key == GLFW_KEY_RIGHT_SUPER) {
        return (action == GLFW_RELEASE) ? mods & (~GLFW_MOD_SUPER) : mods | GLFW_MOD_SUPER;
    }
    return mods;
}

static std::string get_message_id_name(GLuint id) {
    if (id == 0x0500) {
        return "GL_INVALID_ENUM";
    }
    if (id == 0x0501) {
        return "GL_INVALID_VALUE";
    }
    if (id == 0x0502) {
        return "GL_INVALID_OPERATION";
    }
    if (id == 0x0503) {
        return "GL_STACK_OVERFLOW";
    }
    if (id == 0x0504) {
        return "GL_STACK_UNDERFLOW";
    }
    if (id == 0x0505) {
        return "GL_OUT_OF_MEMORY";
    }
    if (id == 0x0506) {
        return "GL_INVALID_FRAMEBUFFER_OPERATION";
    }
    if (id == 0x0507) {
        return "GL_CONTEXT_LOST";
    }
    if (id == 0x8031) {
        return "GL_TABLE_TOO_LARGE";
    }

    return std::to_string(id);
}

#ifdef _WIN32
static std::string GetStack() {
    unsigned int   i;
    void         * stack[100];
    unsigned short frames;
    SYMBOL_INFO  * symbol;
    HANDLE         process;
    std::stringstream output;

    process = GetCurrentProcess();

    SymSetOptions(SYMOPT_LOAD_LINES);

    SymInitialize(process, NULL, TRUE);

    frames = CaptureStackBackTrace(0, 200, stack, NULL);
    symbol = (SYMBOL_INFO *)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
    symbol->MaxNameLen = 255;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

    for (i = 0; i < frames; i++) {
        SymFromAddr(process, (DWORD64)(stack[i]), 0, symbol);
        DWORD  dwDisplacement;
        IMAGEHLP_LINE64 line;

        line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
        if (!strstr(symbol->Name, "khr::getStack") &&
            !strstr(symbol->Name, "khr::DebugCallback") &&
            SymGetLineFromAddr64(process, (DWORD64)(stack[i]), &dwDisplacement, &line)) {

            output << "function: " << symbol->Name <<
                " - line: " << line.LineNumber << "\n";

        }
        if (0 == strcmp(symbol->Name, "main"))
            break;
    }

    free(symbol);
    return output.str();
}
#endif

static void APIENTRY opengl_debug_message_callback(GLenum source, GLenum type, GLuint id, GLenum severity,
    GLsizei length, const GLchar* message, const void* userParam) {
    /* Message Sources
        Source enum                      Generated by
        GL_DEBUG_SOURCE_API              Calls to the OpenGL API
        GL_DEBUG_SOURCE_WINDOW_SYSTEM    Calls to a window - system API
        GL_DEBUG_SOURCE_SHADER_COMPILER  A compiler for a shading language
        GL_DEBUG_SOURCE_THIRD_PARTY      An application associated with OpenGL
        GL_DEBUG_SOURCE_APPLICATION      Generated by the user of this application
        GL_DEBUG_SOURCE_OTHER            Some source that isn't one of these
    */
    /* Message Types
        Type enum                          Meaning
        GL_DEBUG_TYPE_ERROR                An error, typically from the API
        GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR  Some behavior marked deprecated has been used
        GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR   Something has invoked undefined behavior
        GL_DEBUG_TYPE_PORTABILITY          Some functionality the user relies upon is not portable
        GL_DEBUG_TYPE_PERFORMANCE          Code has triggered possible performance issues
        GL_DEBUG_TYPE_MARKER               Command stream annotation
        GL_DEBUG_TYPE_PUSH_GROUP           Group pushing
        GL_DEBUG_TYPE_POP_GROUP            foo
        GL_DEBUG_TYPE_OTHER                Some type that isn't one of these
    */
    /* Message Severity
        Severity enum                    Meaning
        GL_DEBUG_SEVERITY_HIGH           All OpenGL Errors, shader compilation / linking errors, or highly
                                         - dangerous undefined behavior
        GL_DEBUG_SEVERITY_MEDIUM         Major performance warnings, shader compilation / linking
                                         warnings, or the use of deprecated functionality
        GL_DEBUG_SEVERITY_LOW            Redundant state change
                                         performance warning, or unimportant undefined behavior
        GL_DEBUG_SEVERITY_NOTIFICATION   Anything that isn't an
                                         error or performance issue.
    */
    // if (source == GL_DEBUG_SOURCE_API || source == GL_DEBUG_SOURCE_SHADER_COMPILER)
    //    if (type == GL_DEBUG_TYPE_ERROR || type == GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR ||
    //        type == GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR)
    //        if (severity == GL_DEBUG_SEVERITY_HIGH || severity == GL_DEBUG_SEVERITY_MEDIUM)
    //            std::cout << "OpenGL Error: " << message << " (" << get_message_id_name(id) << ")" << std::endl;

    std::string sourceText, typeText, severityText;
    switch (source) {
    case GL_DEBUG_SOURCE_API:
        sourceText = "API";
        break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        sourceText = "Window System";
        break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
        sourceText = "Shader Compiler";
        break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:
        sourceText = "Third Party";
        break;
    case GL_DEBUG_SOURCE_APPLICATION:
        sourceText = "Application";
        break;
    case GL_DEBUG_SOURCE_OTHER:
        sourceText = "Other";
        break;
    default:
        sourceText = "Unknown";
        break;
    }
    switch (type) {
    case GL_DEBUG_TYPE_ERROR:
        typeText = "Error";
        break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        typeText = "Deprecated Behavior";
        break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        typeText = "Undefined Behavior";
        break;
    case GL_DEBUG_TYPE_PORTABILITY:
        typeText = "Portability";
        break;
    case GL_DEBUG_TYPE_PERFORMANCE:
        typeText = "Performance";
        break;
    case GL_DEBUG_TYPE_MARKER:
        typeText = "Marker";
        break;
    case GL_DEBUG_TYPE_PUSH_GROUP:
        typeText = "Push Group";
        break;
    case GL_DEBUG_TYPE_POP_GROUP:
        typeText = "Pop Group";
        break;
    case GL_DEBUG_TYPE_OTHER:
        typeText = "Other";
        break;
    default:
        typeText = "Unknown";
        break;
    }
    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        severityText = "High";
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        severityText = "Medium";
        break;
    case GL_DEBUG_SEVERITY_LOW:
        severityText = "Low";
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        severityText = "Notification";
        break;
    default:
        severityText = "Unknown";
        break;
    }

    std::stringstream output;
    output << "[" << sourceText << " " << severityText << "] (" << typeText << " " << id << " [" << get_message_id_name(id) << "]) " << message << std::endl
           << "stack trace:" << std::endl;
#ifdef _WIN32
    output << GetStack() << std::endl;
    OutputDebugStringA(output.str().c_str());
#else
#ifdef MEGAMOL_USE_BOOST_STACKTRACE
    output << boost::stacktrace::stacktrace() << std::endl;
#else
    output << "(disabled)" << std::endl;
#endif
#endif

    if (type == GL_DEBUG_TYPE_ERROR) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("%s", output.str().c_str());
    } else if (type == GL_DEBUG_TYPE_OTHER || type == GL_DEBUG_TYPE_MARKER) {
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("%s", output.str().c_str());
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("%s", output.str().c_str());
    }
}

void megamol::frontend_resources::WindowManipulation::set_window_title(const char* title) const {
    glfwSetWindowTitle(reinterpret_cast<GLFWwindow*>(window_ptr), title);
}

void megamol::frontend_resources::WindowManipulation::set_framebuffer_size(const unsigned int width, const unsigned int height) const {
    auto window = reinterpret_cast<GLFWwindow*>(this->window_ptr);

    int fbo_width = 0, fbo_height = 0;
    int window_width = 0, window_height = 0;

    glfwSetWindowSizeLimits(window, GLFW_DONT_CARE, GLFW_DONT_CARE, GLFW_DONT_CARE, GLFW_DONT_CARE);
    glfwSetWindowSize(window, width, height);
    glfwGetFramebufferSize(window, &fbo_width, &fbo_height);
    glfwGetWindowSize(window, &window_width, &window_height);

    if (fbo_width != width || fbo_height != height) {
        log("WindowManipulation::set_framebuffer_size(): forcing window size limits to " + std::to_string(width) + "x" + std::to_string(height) + ". You wont be able to resize the window manually after this.");
        glfwSetWindowSizeLimits(window, width, height, width, height);
        glfwSetWindowSize(window, width, height);
        glfwGetFramebufferSize(window, &fbo_width, &fbo_height);
        glfwGetWindowSize(window, &window_width, &window_height);
    }

    if(fbo_width != width || fbo_height != height) {
        log_error("WindowManipulation::set_framebuffer_size() could not enforce window size to achieve requested framebuffer size of w: "
            + std::to_string(width) + ", h: " + std::to_string(height)
            + ".\n Framebuffer has size w: " + std::to_string(fbo_width) + ", h: " + std::to_string(fbo_height)
            + "\n Requesting shutdown.");
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        static_cast<megamol::frontend::OpenGL_GLFW_Service*>(glfwGetWindowUserPointer(window))->setShutdown();
    }
}

void megamol::frontend_resources::WindowManipulation::set_window_position(const unsigned int width, const unsigned int height) const {
    glfwSetWindowPos(reinterpret_cast<GLFWwindow*>(window_ptr), width, height);
}

void megamol::frontend_resources::WindowManipulation::set_swap_interval(const unsigned int wait_frames) const {
    glfwSwapInterval(wait_frames);
}

void megamol::frontend_resources::WindowManipulation::set_fullscreen(const Fullscreen action) const {
    switch (action) {
        case Fullscreen::Maximize:
            glfwMaximizeWindow(reinterpret_cast<GLFWwindow*>(window_ptr));
            break;
        case Fullscreen::Restore:
            glfwRestoreWindow(reinterpret_cast<GLFWwindow*>(window_ptr));
            break;
        default:
            break;
    }
}

namespace megamol {
namespace frontend {

void OpenGL_GLFW_Service::OpenGL_Context::activate() const {
    if (!ptr) return;

    glfwMakeContextCurrent(static_cast<GLFWwindow*>(ptr));
}

void OpenGL_GLFW_Service::OpenGL_Context::close() const {
    if (!ptr) return;

    glfwMakeContextCurrent(nullptr);
}


struct OpenGL_GLFW_Service::PimplData {
    GLFWwindow* glfwContextWindowPtr{nullptr};
    OpenGL_GLFW_Service::Config config;    // keep copy of user-provided config
    std::string fullWindowTitle;
    std::chrono::system_clock::time_point last_time;
    megamol::frontend_resources::FrameStatistics* frame_statistics{nullptr};
    std::array<GLFWcursor*, 9> mouse_cursors;
};

OpenGL_GLFW_Service::~OpenGL_GLFW_Service() {
}

bool OpenGL_GLFW_Service::init(void* configPtr) {
    if (configPtr == nullptr) return false;

    return init(*static_cast<Config*>(configPtr));
}

bool OpenGL_GLFW_Service::init(const Config& config) {
    m_pimpl = std::unique_ptr<PimplData, std::function<void(PimplData*)>>(new PimplData, [](PimplData* ptr) { delete ptr; });
    if (!m_pimpl) {
        log_error("could not allocate private data");
        return false;
    }
    m_pimpl->config = config;

    glfwSetErrorCallback(glfw_error_callback);

    bool success_glfw = glfwInit();
    if (!success_glfw) {
        log_error("could not initialize GLFW for OpenGL window. \nmaybe your machine is using outdated graphics hardware, drivers or you are working remotely?");
        return false; // glfw had error on init; abort
    }

    // NOTE: most of the following GLFW/GL setup code is ported from the mmconsole Window class
    // it may be outdated or implement features no longer required by MegaMol

    // init glfw window and OpenGL Context
    ::glfwWindowHint(GLFW_ALPHA_BITS, 8);
    ::glfwWindowHint(GLFW_DECORATED,
        (m_pimpl->config.windowPlacement.fullScreen) ? (GL_FALSE) : (m_pimpl->config.windowPlacement.noDec ? GL_FALSE : GL_TRUE));
    ::glfwWindowHint(GLFW_VISIBLE, GL_FALSE); // initially invisible

    int monCnt = 0;
    GLFWmonitor** monitors = ::glfwGetMonitors(&monCnt); // primary monitor is first in list
    if (!monitors) return false;                         // no monitor found; abort

    // in fullscreen, use last available monitor as to not block primary monitor, where the user may have important
    // stuff he wants to look at
    int monitorNr =
        (m_pimpl->config.windowPlacement.fullScreen)
            ? std::max<int>(0, std::min<int>(monCnt - 1,
                                   m_pimpl->config.windowPlacement.mon)) // if fullscreen, use last or user-provided monitor
            : (0);                                              // if windowed, use primary monitor
    GLFWmonitor* selectedMonitor = monitors[monitorNr];
    if (!selectedMonitor) return false; // selected monitor not valid for some reason; abort

    const GLFWvidmode* mode = ::glfwGetVideoMode(selectedMonitor);
    if (!mode) return false; // error while receiving monitor mode; abort

    // window size for windowed mode
    int initial_width = 0;
    int initial_height = 0;
    if (!m_pimpl->config.windowPlacement.fullScreen) {
        if (m_pimpl->config.windowPlacement.size && (m_pimpl->config.windowPlacement.w > 0) && (m_pimpl->config.windowPlacement.h > 0)) {
            initial_width = m_pimpl->config.windowPlacement.w;
            initial_height = m_pimpl->config.windowPlacement.h;
        } else {
            log("No useful window size given. Making one up");
            // no useful window size given, derive one from monitor resolution
            initial_width = mode->width * 3 / 4;
            initial_height = mode->height * 3 / 4;
        }
    }

    if(m_pimpl->config.windowPlacement.topMost){
        ::glfwWindowHint(GLFW_FLOATING, GL_TRUE); // floating above other windows / top most
    }

    // options for fullscreen mode
    if (m_pimpl->config.windowPlacement.fullScreen) {
        if (m_pimpl->config.windowPlacement.pos)
            log("Ignoring window placement position when requesting fullscreen.");

        if (m_pimpl->config.windowPlacement.size &&
            ((m_pimpl->config.windowPlacement.w != mode->width) || (m_pimpl->config.windowPlacement.h != mode->height)))
            log("Changing screen resolution is currently not supported.");

        if (m_pimpl->config.windowPlacement.noDec)
            log("Ignoring no-decorations setting when requesting fullscreen.");

        /* note we do not use a real fullscrene mode, since then we would have focus-iconify problems */
        initial_width = mode->width;
        initial_height = mode->height;

        ::glfwWindowHint(GLFW_RED_BITS, mode->redBits);
        ::glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
        ::glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
        ::glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
        // this only works since we are NOT setting a monitor
        ::glfwWindowHint(GLFW_FLOATING, GL_TRUE); // floating above other windows / top most

        // will place 'fullscreen' window at origin of monitor
        int mon_x, mon_y;
        ::glfwGetMonitorPos(selectedMonitor, &mon_x, &mon_y);
        m_pimpl->config.windowPlacement.x = mon_x;
        m_pimpl->config.windowPlacement.y = mon_y;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, m_pimpl->config.versionMajor);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, m_pimpl->config.versionMinor);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, m_pimpl->config.enableKHRDebug ? GLFW_TRUE : GLFW_FALSE);
    // context profiles available since 3.2
    bool has_profiles = m_pimpl->config.versionMajor > 3 || m_pimpl->config.versionMajor == 3 && m_pimpl->config.versionMinor >= 2;
    if (has_profiles)
        glfwWindowHint(GLFW_OPENGL_PROFILE, m_pimpl->config.glContextCoreProfile ? GLFW_OPENGL_CORE_PROFILE : GLFW_OPENGL_COMPAT_PROFILE);

    std::string profile_name =
        has_profiles
            ? ((m_pimpl->config.glContextCoreProfile ? "Core" : "Compatibility") + std::string(" Profile"))
            : "";

    log("Requesting OpenGL "
        + std::to_string(m_pimpl->config.versionMajor) + "."
        + std::to_string(m_pimpl->config.versionMinor) + " "
        + profile_name
        + (m_pimpl->config.enableKHRDebug ? ", Debug Context" : "" )
    );

    auto& window_ptr = m_pimpl->glfwContextWindowPtr;
    window_ptr = ::glfwCreateWindow(initial_width, initial_height,
        m_pimpl->config.windowTitlePrefix.c_str(), nullptr, nullptr);
    m_opengl_context_impl.ptr = window_ptr;

    if (!window_ptr) {
        log_error("Could not create GLFW Window. You probably do not have OpenGL support. Your graphics hardware might "
                "be very old, your drivers could be outdated or you are running in a remote desktop session.");
        return false;
    }
    log("Create window with size w: " + std::to_string(initial_width) + " h: " + std::to_string(initial_height));


    // we publish a fake GL context to have a resource others can ask for
    // however, we set the actual GL context active for the main thread and leave it active until further design requirements arise
    m_opengl_context = &m_fake_opengl_context;
    ::glfwMakeContextCurrent(window_ptr);
    //m_opengl_context_impl.activate();

    //if(gladLoadGLLoader((GLADloadproc) glfwGetProcAddress) == 0) {
    if(gladLoadGL() == 0) {
        log_error("Failed to load OpenGL functions via glad");
        return false;
    }
#ifdef _WIN32
    if (gladLoadWGL(wglGetCurrentDC()) == 0) {
        log_error("Failed to load OpenGL WGL functions via glad");
        return false;
    }
#else
    Display* display = XOpenDisplay(NULL);
    gladLoadGLX(display, DefaultScreen(display));
    XCloseDisplay(display);
#endif

    log(std::string("OpenGL Context Info")
        + "\n\tVersion:  " + reinterpret_cast<const char*>(glGetString(GL_VERSION))
        + "\n\tVendor:   " + reinterpret_cast<const char*>(glGetString(GL_VENDOR))
        + "\n\tRenderer: " + reinterpret_cast<const char*>(glGetString(GL_RENDERER))
        + "\n\tGLSL:     " + reinterpret_cast<const char*>(glGetString(GL_SHADING_LANGUAGE_VERSION))
    );

    if (m_pimpl->config.enableKHRDebug) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
                GLuint ignorethis;
                ignorethis = 131185;
                glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DONT_CARE, 1, &ignorethis, GL_FALSE);
                ignorethis = 131184;
                glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DONT_CARE, 1, &ignorethis, GL_FALSE);
                ignorethis = 131204;
                glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DONT_CARE, 1, &ignorethis, GL_FALSE);
        glDebugMessageCallback(opengl_debug_message_callback, nullptr);
        log("Enabled OpenGL debug context. Will print debug messages.");
    }

    if (config.windowPlacement.noCursor) {
        ::glfwSetInputMode(window_ptr, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    } else {
        create_glfw_mouse_cursors();
    }

    // note the m_data window position got overwritten with monitor position for fullscreen mode
    if (m_pimpl->config.windowPlacement.pos || m_pimpl->config.windowPlacement.fullScreen)
        ::glfwSetWindowPos(
            window_ptr, m_pimpl->config.windowPlacement.x, m_pimpl->config.windowPlacement.y);

    register_glfw_callbacks();

    int vsync = (m_pimpl->config.enableVsync) ? 1 : 0;
    ::glfwSwapInterval(vsync);

    ::glfwShowWindow(window_ptr);
    //::glfwMakeContextCurrent(nullptr);

    m_windowEvents._clipboard_user_data = window_ptr;
    m_windowEvents._getClipboardString_Func = [](void* user_data) -> const char* {
        return glfwGetClipboardString(reinterpret_cast<GLFWwindow*>(user_data));
    };
    m_windowEvents._setClipboardString_Func = [](void* user_data, const char* string) {
        glfwSetClipboardString(reinterpret_cast<GLFWwindow*>(user_data), string);
    };
    m_windowEvents.previous_state.time = glfwGetTime();

    m_windowManipulation.window_ptr = window_ptr;

    m_windowManipulation.set_mouse_cursor = [&](const int cursor_id) -> void {
        update_glfw_mouse_cursors(cursor_id);
    };

    // make the events and resources managed/provided by this service available to the outside world
    m_renderResourceReferences = {
        {"KeyboardEvents", m_keyboardEvents},
        {"MouseEvents", m_mouseEvents},
        {"WindowEvents", m_windowEvents},
        {"FramebufferEvents", m_framebufferEvents},
        {"IOpenGL_Context", *m_opengl_context},
        {"WindowManipulation", m_windowManipulation}
    };

    m_requestedResourcesNames = {
          "FrameStatistics"
    };

    m_pimpl->last_time = std::chrono::system_clock::now();

    log("initialized successfully");
    return true;
}

void OpenGL_GLFW_Service::do_every_second() {
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    auto& last_time = m_pimpl->last_time;

    if (now - last_time > std::chrono::seconds(1)) {
        last_time = now;

        auto const cut_off = [](std::string const& s) -> std::string {
            const auto found_dot = s.find_first_of(".,");
            const auto substr = s.substr(0, found_dot + 1 + 1);
            return substr;
        };

        std::string fps = std::to_string(m_pimpl->frame_statistics->last_averaged_fps);
        std::string mspf = std::to_string(m_pimpl->frame_statistics->last_averaged_mspf);
        std::string title =
            m_pimpl->config.windowTitlePrefix
            + " [" + cut_off(fps) + "f/s, " + cut_off(mspf) + "ms/f]";
        glfwSetWindowTitle(m_pimpl->glfwContextWindowPtr, title.c_str());

#ifdef _WIN32
        // TODO fix this for EGL + Win
        //log("Periodic reordering of windows.");
        if (m_pimpl->config.windowPlacement.topMost) {
            SetWindowPos(glfwGetWin32Window(m_pimpl->glfwContextWindowPtr), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE);
        }
#endif

    }
}

#define that static_cast<OpenGL_GLFW_Service*>(::glfwGetWindowUserPointer(wnd))
void OpenGL_GLFW_Service::register_glfw_callbacks() {
    auto& window_ptr = m_pimpl->glfwContextWindowPtr;

    ::glfwSetWindowUserPointer(window_ptr, this); // this is ok, as long as no one derives from this class

    // keyboard events
    ::glfwSetKeyCallback(window_ptr, [](GLFWwindow* wnd, int key, int scancode, int action, int mods){
        // Fix mods, see: https://github.com/glfw/glfw/issues/1630
        mods = fixGlfwKeyboardMods(mods, key, action);
        that->glfw_onKey_func(key, scancode, action, mods);
    });

    ::glfwSetCharCallback(window_ptr, [](GLFWwindow* wnd, unsigned int codepoint) {
        that->glfw_onChar_func(codepoint);
    });
    // this->m_keyboardEvents; // ignore because no interaction happened yet

    // mouse events
    ::glfwSetMouseButtonCallback(window_ptr, [](GLFWwindow* wnd, int button, int action, int mods) {
        that->glfw_onMouseButton_func(button, action, mods);
    });

    ::glfwSetCursorPosCallback(window_ptr, [](GLFWwindow* wnd, double xpos, double ypos) {
        // cursor (x,y) position in screen coordinates relative to upper-left corner
        that->glfw_onMouseCursorPosition_func(xpos, ypos);
    });

    ::glfwSetCursorEnterCallback(window_ptr, [](GLFWwindow* wnd, int entered) {
        that->glfw_onMouseCursorEnter_func(entered == GLFW_TRUE);
    });

    ::glfwSetScrollCallback(window_ptr, [](GLFWwindow* wnd, double xoffset, double yoffset) {
        that->glfw_onMouseScroll_func(xoffset, yoffset);
    });

    // set current state for mouse events
    // this->m_mouseEvents.previous_state.buttons; // ignore because no interaction yet
    this->m_mouseEvents.previous_state.entered = glfwGetWindowAttrib(window_ptr, GLFW_HOVERED);
    ::glfwGetCursorPos(window_ptr, &this->m_mouseEvents.previous_state.x_cursor_position,
        &this->m_mouseEvents.previous_state.y_cursor_position);
    this->m_mouseEvents.previous_state.x_scroll = 0.0;
    this->m_mouseEvents.previous_state.y_scroll = 0.0;

    // window events
    ::glfwSetWindowSizeCallback(window_ptr, [](GLFWwindow* wnd, int width /* in screen coordinates of the window */, int height) {
        that->glfw_onWindowSize_func(width, height);
    });
    ::glfwSetWindowFocusCallback(window_ptr, [](GLFWwindow* wnd, int focused) { that->glfw_onWindowFocus_func(focused == GLFW_TRUE); });
    ::glfwSetWindowCloseCallback(window_ptr, [](GLFWwindow* wnd) { that->glfw_onWindowShouldClose_func(true); });
    ::glfwSetWindowIconifyCallback(window_ptr, [](GLFWwindow* wnd, int iconified) { that->glfw_onWindowIconified_func(iconified == GLFW_TRUE); });

    ::glfwSetWindowContentScaleCallback(window_ptr, [](GLFWwindow* wnd, float xscale, float yscale) {
        that->glfw_onWindowContentScale_func(xscale, yscale);
    });

    ::glfwSetDropCallback(window_ptr, [](GLFWwindow* wnd, int path_count, const char* paths[]) {
        that->glfw_onPathDrop_func(path_count, paths);
    });
    // void outer_glfw_WindowPosition_func(GLFWwindow* wnd, int xpos, int ypos) { that->glfw_WindowPosition_func(xpos, ypos); }

    // set current window state, needed for correct linux window management with imgui
    glfwGetWindowSize(
        window_ptr, &this->m_windowEvents.previous_state.width, &this->m_windowEvents.previous_state.height);
    this->m_windowEvents.previous_state.is_focused = (GLFW_TRUE == glfwGetWindowAttrib(window_ptr, GLFW_FOCUSED));
    this->m_windowEvents.previous_state.is_iconified =
        (GLFW_TRUE == glfwGetWindowAttrib(window_ptr, GLFW_ICONIFIED));
    this->m_windowEvents.previous_state.should_close = (GLFW_TRUE == glfwWindowShouldClose(window_ptr));
    glfwGetWindowContentScale(window_ptr,
        &this->m_windowEvents.previous_state.x_contentscale,
        &this->m_windowEvents.previous_state.y_contentscale);

    glfw_onWindowSize_func(
        this->m_windowEvents.previous_state.width,
        this->m_windowEvents.previous_state.height);
    glfw_onWindowFocus_func(this->m_windowEvents.previous_state.is_focused);
    glfw_onWindowIconified_func(this->m_windowEvents.previous_state.is_iconified);
    glfw_onWindowContentScale_func(
        this->m_windowEvents.previous_state.x_contentscale,
        this->m_windowEvents.previous_state.y_contentscale);

    // set callbacks
    ::glfwSetFramebufferSizeCallback(window_ptr, [](GLFWwindow* wnd, int widthpx, int heightpx) {
        that->glfw_onFramebufferSize_func(widthpx, heightpx);
    });

    // set current framebuffer state as pending event
    glfwGetFramebufferSize(window_ptr,
        &this->m_framebufferEvents.previous_state.width,
        &this->m_framebufferEvents.previous_state.height);

    glfw_onFramebufferSize_func(
        this->m_framebufferEvents.previous_state.width,
        this->m_framebufferEvents.previous_state.height);
}
#undef that

void OpenGL_GLFW_Service::close() {
    if (!m_pimpl) // this GLFW context service is not initialized
        return;

    ::glfwMakeContextCurrent(m_pimpl->glfwContextWindowPtr);

    for (auto& mouse_cursor_ptr : m_pimpl->mouse_cursors) {
        ::glfwDestroyCursor(mouse_cursor_ptr);
        mouse_cursor_ptr = nullptr;
    }

    // GL context and destruction of all other things happens in destructors of pimpl data members
    if (m_pimpl->glfwContextWindowPtr) ::glfwDestroyWindow(m_pimpl->glfwContextWindowPtr);
    m_pimpl->glfwContextWindowPtr = nullptr;
    this->m_pimpl.release();

    ::glfwMakeContextCurrent(nullptr);
    ::glfwTerminate();
}

void OpenGL_GLFW_Service::updateProvidedResources() {
    // poll events for all GLFW windows shared by this context. this also issues the callbacks.
    // note at this point there is no GL context active.
    // event struct get filled via GLFW callbacks when new input events come in during glfwPollEvents()
    ::glfwPollEvents(); // may only be called from main thread

    m_windowEvents.time = glfwGetTime();
    // from GLFW Docs:
    // Do not assume that callbacks will only be called through glfwPollEvents().
    // While it is necessary to process events in the event queue,
    // some window systems will send some events directly to the application,
    // which in turn causes callbacks to be called outside of regular event processing.

    auto& should_close_events = this->m_windowEvents.should_close_events;
    if (should_close_events.size() && std::count(should_close_events.begin(), should_close_events.end(), true))
        this->setShutdown(true); // cleanup of this service and dependent GL stuff is triggered via this shutdown hint
}

void OpenGL_GLFW_Service::digestChangedRequestedResources() {
    // we dont depend on outside resources
}

void OpenGL_GLFW_Service::resetProvidedResources() {
    m_keyboardEvents.clear();
    m_mouseEvents.clear();
    m_windowEvents.clear();
    m_framebufferEvents.clear();
}

void OpenGL_GLFW_Service::preGraphRender() {
    // if (window_ptr == nullptr) return;

    // e.g. start frame timer

    // rendering via MegaMol View is called after this function finishes
    // in the end this calls the equivalent of ::mmcRenderView(hView, &renderContext)
    // which leads to view.Render()
}

void OpenGL_GLFW_Service::postGraphRender() {
    // end frame timer
    // update window name

    ::glfwSwapBuffers(m_pimpl->glfwContextWindowPtr);

    //::glfwMakeContextCurrent(window_ptr);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | GL_ACCUM_BUFFER_BIT);
    //::glfwMakeContextCurrent(nullptr);

    do_every_second();
}

std::vector<FrontendResource>& OpenGL_GLFW_Service::getProvidedResources() {
    return m_renderResourceReferences;
}

const std::vector<std::string> OpenGL_GLFW_Service::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

void OpenGL_GLFW_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    m_requestedResourceReferences = resources;

    m_pimpl->frame_statistics = &const_cast<megamol::frontend_resources::FrameStatistics&>(resources[0].getResource<megamol::frontend_resources::FrameStatistics>());
}

void OpenGL_GLFW_Service::glfw_onKey_func(const int key, const int scancode, const int action, const int mods) {

    frontend_resources::Key key_ = static_cast<frontend_resources::Key>(key);
    frontend_resources::KeyAction action_(frontend_resources::KeyAction::RELEASE);
    switch (action) {
    case GLFW_PRESS:
        action_ = frontend_resources::KeyAction::PRESS;
        break;
    case GLFW_REPEAT:
        action_ = frontend_resources::KeyAction::REPEAT;
        break;
    case GLFW_RELEASE:
        action_ = frontend_resources::KeyAction::RELEASE;
        break;
    }

    frontend_resources::Modifiers mods_;
    if ((mods & GLFW_MOD_SHIFT) == GLFW_MOD_SHIFT) mods_ |= frontend_resources::Modifier::SHIFT;
    if ((mods & GLFW_MOD_CONTROL) == GLFW_MOD_CONTROL) mods_ |= frontend_resources::Modifier::CTRL;
    if ((mods & GLFW_MOD_ALT) == GLFW_MOD_ALT) mods_ |= frontend_resources::Modifier::ALT;

    this->m_keyboardEvents.key_events.emplace_back(std::make_tuple(key_, action_, mods_));
}

void OpenGL_GLFW_Service::glfw_onChar_func(const unsigned int codepoint) {
    this->m_keyboardEvents.codepoint_events.emplace_back(codepoint);
}

void OpenGL_GLFW_Service::glfw_onMouseCursorPosition_func(const double xpos, const double ypos) {

    this->m_mouseEvents.position_events.emplace_back(std::make_tuple(xpos, ypos));
}

void OpenGL_GLFW_Service::glfw_onMouseButton_func(const int button, const int action, const int mods) {
    frontend_resources::MouseButton btn = static_cast<frontend_resources::MouseButton>(button);
    frontend_resources::MouseButtonAction btnaction =
        (action == GLFW_PRESS) ? frontend_resources::MouseButtonAction::PRESS : frontend_resources::MouseButtonAction::RELEASE;

    frontend_resources::Modifiers btnmods;
    if ((mods & GLFW_MOD_SHIFT) == GLFW_MOD_SHIFT) btnmods |= frontend_resources::Modifier::SHIFT;
    if ((mods & GLFW_MOD_CONTROL) == GLFW_MOD_CONTROL) btnmods |= frontend_resources::Modifier::CTRL;
    if ((mods & GLFW_MOD_ALT) == GLFW_MOD_ALT) btnmods |= frontend_resources::Modifier::ALT;

    this->m_mouseEvents.buttons_events.emplace_back(std::make_tuple(btn, btnaction, btnmods));
}

void OpenGL_GLFW_Service::glfw_onMouseScroll_func(const double xoffset, const double yoffset) {
    this->m_mouseEvents.scroll_events.emplace_back(std::make_tuple(xoffset, yoffset));
}

void OpenGL_GLFW_Service::glfw_onMouseCursorEnter_func(const bool entered) {
    this->m_mouseEvents.enter_events.emplace_back(entered);
}

void OpenGL_GLFW_Service::glfw_onFramebufferSize_func(const int widthpx, const int heightpx) {
    this->m_framebufferEvents.size_events.emplace_back(frontend_resources::FramebufferState{widthpx, heightpx});
}

void OpenGL_GLFW_Service::glfw_onWindowSize_func(const int width, const int height) { // in screen coordinates, of the window
    this->m_windowEvents.size_events.emplace_back(std::tuple(width, height));
}

void OpenGL_GLFW_Service::glfw_onWindowFocus_func(const bool focused) {
    this->m_windowEvents.is_focused_events.emplace_back(focused);
}

void OpenGL_GLFW_Service::glfw_onWindowShouldClose_func(const bool shouldclose) {
    this->m_windowEvents.should_close_events.emplace_back(shouldclose);
}

void OpenGL_GLFW_Service::glfw_onWindowIconified_func(const bool iconified) {
    this->m_windowEvents.is_iconified_events.emplace_back(iconified);
}

void OpenGL_GLFW_Service::glfw_onWindowContentScale_func(const float xscale, const float yscale) {
    this->m_windowEvents.content_scale_events.emplace_back(std::tuple(xscale, yscale));
}

void OpenGL_GLFW_Service::glfw_onPathDrop_func(const int path_count, const char* paths[]) {
    std::vector<std::string> paths_;
    paths_.reserve(path_count);

    for (int i = 0; i < path_count; i++)
        paths_.emplace_back(std::string(paths[i]));

    this->m_windowEvents.dropped_path_events.push_back(paths_);
}

void OpenGL_GLFW_Service::create_glfw_mouse_cursors(void) {
    // See imgui_impl_glfw.cpp for reference
    for (auto& mouse_cursor_ptr : m_pimpl->mouse_cursors) {
        mouse_cursor_ptr = nullptr;
    }
    if (m_pimpl->mouse_cursors.size() != 9) {
        return;
    }
    // See imgui.h: enum ImGuiMouseCursor_ 
    GLFWerrorfun prev_error_callback = ::glfwSetErrorCallback(nullptr);
    m_pimpl->mouse_cursors[0] = ::glfwCreateStandardCursor(GLFW_ARROW_CURSOR); // ImGuiMouseCursor_Arrow
    m_pimpl->mouse_cursors[1] = ::glfwCreateStandardCursor(GLFW_IBEAM_CURSOR); // ImGuiMouseCursor_TextInput
    m_pimpl->mouse_cursors[3] = ::glfwCreateStandardCursor(GLFW_VRESIZE_CURSOR); // ImGuiMouseCursor_ResizeNS
    m_pimpl->mouse_cursors[4] = ::glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR); // ImGuiMouseCursor_ResizeEW
    m_pimpl->mouse_cursors[7] = ::glfwCreateStandardCursor(GLFW_HAND_CURSOR);    // ImGuiMouseCursor_Hand
#if ((GLFW_VERSION_MAJOR * 1000 + GLFW_VERSION_MINOR * 100) >= 3400) // glfw 3.4+
    m_pimpl->mouse_cursors[2] = ::glfwCreateStandardCursor(GLFW_RESIZE_ALL_CURSOR); // ImGuiMouseCursor_ResizeAll
    m_pimpl->mouse_cursors[5] = ::glfwCreateStandardCursor(GLFW_RESIZE_NESW_CURSOR); // ImGuiMouseCursor_ResizeNESW
    m_pimpl->mouse_cursors[6] = ::glfwCreateStandardCursor(GLFW_RESIZE_NWSE_CURSOR); // ImGuiMouseCursor_ResizeNWSE
    m_pimpl->mouse_cursors[8] = ::glfwCreateStandardCursor(GLFW_NOT_ALLOWED_CURSOR); // ImGuiMouseCursor_NotAllowed
#else
    m_pimpl->mouse_cursors[2] = ::glfwCreateStandardCursor(GLFW_ARROW_CURSOR); // ImGuiMouseCursor_ResizeAll
    m_pimpl->mouse_cursors[5] = ::glfwCreateStandardCursor(GLFW_ARROW_CURSOR); // ImGuiMouseCursor_ResizeNESW
    m_pimpl->mouse_cursors[6] = ::glfwCreateStandardCursor(GLFW_ARROW_CURSOR); // ImGuiMouseCursor_ResizeNWSE
    m_pimpl->mouse_cursors[8] = ::glfwCreateStandardCursor(GLFW_ARROW_CURSOR); // ImGuiMouseCursor_NotAllowed
#endif
    ::glfwSetErrorCallback(prev_error_callback);
}

void OpenGL_GLFW_Service::update_glfw_mouse_cursors(const int cursor_id) {
    if (m_pimpl->config.windowPlacement.noCursor) {
        ::glfwSetInputMode(m_pimpl->glfwContextWindowPtr, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        return;
    }

    if ((cursor_id >= 0) && (cursor_id < m_pimpl->mouse_cursors.size())) {
        ::glfwSetCursor(m_pimpl->glfwContextWindowPtr,
            m_pimpl->mouse_cursors[cursor_id] ? m_pimpl->mouse_cursors[cursor_id]
                                              : m_pimpl->mouse_cursors[static_cast<int>(GLFW_ARROW_CURSOR)]);
        ::glfwSetInputMode(m_pimpl->glfwContextWindowPtr, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    } else {
        ::glfwSetInputMode(m_pimpl->glfwContextWindowPtr, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    }
}

// { glfw calls used somewhere by imgui but not covered by this class
// glfwGetWin32Window(g_Window);
//
// glfwGetMouseButton(g_Window, i) != 0;
// glfwGetCursorPos(g_Window, &mouse_x, &mouse_y);
// glfwSetCursorPos(g_Window, (double)mouse_pos_backup.x, (double)mouse_pos_backup.y);
//
// glfwGetInputMode(g_Window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED)
// }

} // namespace frontend
} // namespace megamol
