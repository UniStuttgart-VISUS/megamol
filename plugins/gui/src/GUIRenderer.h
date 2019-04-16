/*
 * GUIRenderer.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

/**
 * TODO
 *
 * - Fix lost keyboard/mouse input for low frame rates
 *
 *
 * USED HOKEYS:
 *
 * - Show/hide Windows: F12 - F9
 * - Reset windows:     Shift + (Window show/hide hotkeys)
 * - Quit program:      Esc, Alt + F4
 *
 */

#ifndef MEGAMOL_GUI_GUIRENDERER_H_INCLUDED
#define MEGAMOL_GUI_GUIRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/TernaryParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/view/CallSplitViewOverlay.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/view/Renderer3DModule.h"

#include "vislib/UTF8Encoder.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <iomanip> // setprecision
#include <sstream> // stringstream


#include "GUITransferFunctionEditor.h"
#include "GUIUtility.h"
#include "GUIWindowManager.h"

#include "imgui_impl_opengl3.h"


namespace megamol {
namespace gui {


template <class M, class C> class GUIRenderer : public M, GUIUtility {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void);

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description(void) { return "Graphical user interface renderer"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable(void) { return true; }

    /**
     * Initialises a new instance.
     */
    GUIRenderer();

    /**
     * Finalises an instance.
     */
    virtual ~GUIRenderer();

protected:
    // TYPES ------------------------------------------------------------------

    /** Type for window configuration */
    typedef megamol::gui::GUIWindowManager::WindowConfiguration GUIWinConfig;

    // FUNCTIONS --------------------------------------------------------------

    virtual bool create() override;

    virtual void release() override;

    virtual bool GetExtents(C& call) override;

    virtual bool Render(C& call) override;

    virtual bool OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) override;

    virtual bool OnChar(unsigned int codePoint) override;

    virtual bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    virtual bool OnMouseMove(double x, double y) override;

    virtual bool OnMouseScroll(double dx, double dy) override;

    /**
     * Callback forwarding OnRender request.
     */
    bool OnOverlayCallback(core::Call& call);

    /**
     * Callback forwarding OnKey request.
     */
    bool OnKeyCallback(core::Call& call);

    /**
     * Callback forwarding OnChar request.
     */
    bool OnCharCallback(core::Call& call);

    /**
     * Callback forwarding OnMouse request.
     */
    bool OnMouseButtonCallback(core::Call& call);

    /**
     * Callback forwarding OnMouseMove request.
     */
    bool OnMouseMoveCallback(core::Call& call);

    /**
     * Callback forwarding OnMouseScroll request.
     */
    bool OnMouseScrollCallback(core::Call& call);

private:
    // ENUMS ------------------------------------------------------------------

    /** ImGui key map assignment for text manipulation hotkeys (using last unused indices < 512) */
    enum GuiTextModHotkeys { CTRL_A = 506, CTRL_C = 507, CTRL_V = 508, CTRL_X = 509, CTRL_Y = 510, CTRL_Z = 511 };

    // VARIABLES --------------------------------------------------------------

    /** The overlay callee slot */
    core::CalleeSlot overlay_slot;

    /** The decorated renderer caller slot */
    core::CallerSlot decorated_renderer_slot;

    /** The ImGui context created and used by this GUIRenderer */
    ImGuiContext* imgui_context;

    /** The window manager. */
    GUIWindowManager window_manager;

    /** The transfer function editor. */
    GUITransferFunctionEditor tf_editor;

    /** Flag indicating that this window configuration profile should be applied. */
    std::string load_new_profile;

    /** File name of font file to load. */
    std::string load_new_font_filename;

    /** Font size of font to load. */
    float load_new_font_size;

    /** Last instance time.  */
    double last_instance_time;

    /** Additional UTF-8 glyph ranges for ImGui fonts. */
    std::vector<ImWchar> font_utf8_ranges;

    // FUNCTIONS --------------------------------------------------------------

    /**
     * Renders GUI.
     *
     * @param viewport      The currently available viewport.
     * @param instanceTime  The current instance time.
     */
    bool render(vislib::math::Rectangle<int> viewport, double instanceTime);

    /**
     * Callback for drawing the parameter window.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawMainWindowCallback(const std::string& window_name, GUIWinConfig& window_config);

    /**
     * Draws parameters and options.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawParametersCallback(const std::string& window_name, GUIWinConfig& window_config);

    /**
     * Draws fps overlay window.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawFpsWindowCallback(const std::string& window_name, GUIWinConfig& window_config);

    /**
     * Callback for drawing font selection window.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawFontWindowCallback(const std::string& window_name, GUIWinConfig& window_config);

    /**
     * Callback for drawing the demo window.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawTFWindowCallback(const std::string& window_name, GUIWinConfig& window_config);

    // ---------------------------------

    /**
     * Draws the menu bar.
     */
    void drawMenu(void);

    /**
     * Draws a parameter.
     *
     * @param mod   Module the paramter belongs to.
     * @param slot  The current parameter slot.
     */
    void drawParameter(const core::Module& mod, core::param::ParamSlot& slot);

    /**
     * Draws only a button parameter's hotkey.
     *
     * @param mod   Module the paramter belongs to.
     * @param slot  The current parameter slot.
     */
    void drawParameterHotkey(const core::Module& mod, core::param::ParamSlot& slot);

    /**
     * Check if module's parameters should be considered.
     */
    bool considerModule(const std::string& modname, std::vector<std::string>& modules_list);

    /**
     * Shutdown megmol program.
     */
    void shutdown(void);

    // ------------------------------------------------------------------------
};


typedef GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D> GUIRenderer2D;
typedef GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D> GUIRenderer3D;


/**
 * GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::GUIRenderer
 */
template <>
inline GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::GUIRenderer()
    : core::view::Renderer2DModule()
    , decorated_renderer_slot("decoratedRenderer", "Connects to another 2D Renderer being decorated")
    , overlay_slot("overlayRender", "Connected with SplitView for special overlay rendering")
    , imgui_context(nullptr)
    , window_manager()
    , tf_editor()
    , load_new_profile()
    , load_new_font_filename()
    , load_new_font_size(13.0f)
    , last_instance_time(0.0)
    , font_utf8_ranges() {

    this->decorated_renderer_slot.SetCompatibleCall<core::view::CallRender2DDescription>();
    this->MakeSlotAvailable(&this->decorated_renderer_slot);

    // InputCall
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::CallSplitViewOverlay::FunctionName(core::view::CallSplitViewOverlay::FnOverlay),
        &GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::OnOverlayCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnKey),
        &GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::OnKeyCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnChar),
        &GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::OnCharCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseButton),
        &GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::OnMouseButtonCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseMove),
        &GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::OnMouseMoveCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseScroll),
        &GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::OnMouseScrollCallback);
    this->MakeSlotAvailable(&this->overlay_slot);
}


/**
 * GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::GUIRenderer
 */
template <>
inline GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::GUIRenderer()
    : core::view::Renderer3DModule()
    , decorated_renderer_slot("decoratedRenderer", "Connects to another 2D Renderer being decorated")
    , overlay_slot("overlayRender", "Connected with SplitView for special overlay rendering")
    , imgui_context(nullptr)
    , window_manager()
    , tf_editor()
    , load_new_profile()
    , load_new_font_filename()
    , load_new_font_size(13.0f)
    , last_instance_time(0.0)
    , font_utf8_ranges() {

    this->decorated_renderer_slot.SetCompatibleCall<core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->decorated_renderer_slot);

    // Overlay Call
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::CallSplitViewOverlay::FunctionName(core::view::CallSplitViewOverlay::FnOverlay),
        &GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::OnOverlayCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnKey),
        &GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::OnKeyCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnChar),
        &GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::OnCharCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseButton),
        &GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::OnMouseButtonCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseMove),
        &GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::OnMouseMoveCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseScroll),
        &GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::OnMouseScrollCallback);
    this->MakeSlotAvailable(&this->overlay_slot);
}


/**
 * GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::ClassName
 */
template <> inline const char* GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::ClassName(void) {

    return "GUIRenderer2D";
}


/**
 * GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::ClassName
 */
template <> inline const char* GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::ClassName(void) {

    return "GUIRenderer3D";
}


/**
 * GUIRenderer<M, C>::~GUIRenderer
 */
template <class M, class C> GUIRenderer<M, C>::~GUIRenderer() { this->Release(); }


/**
 * GUIRenderer<M, C>::create
 */
template <class M, class C> bool GUIRenderer<M, C>::create() {

    // Create ImGui context ---------------------------------------------------
    // Check for existing context and share FontAtlas with new context (required by ImGui).
    bool other_context = (ImGui::GetCurrentContext() != nullptr);
    ImFontAtlas* current_fonts = nullptr;
    if (other_context) {
        ImGuiIO& current_io = ImGui::GetIO();
        current_fonts = current_io.Fonts;
    }
    IMGUI_CHECKVERSION();
    this->imgui_context = ImGui::CreateContext(current_fonts);
    if (this->imgui_context == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[GUIRenderer][create] Couldn't create ImGui context");
        return false;
    }
    ImGui::SetCurrentContext(this->imgui_context);

    // Init OpenGL for ImGui --------------------------------------------------
    const char* glsl_version = "#version 130"; /// "#version 150"
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Register window callbacks in window managerS
    this->window_manager.RegisterDrawWindowCallback(GUIWindowManager::WindowDrawCallback::MAIN,
        [&, this](const std::string& window_name, GUIWindowManager::WindowConfiguration& window_config) {
            this->drawMainWindowCallback(window_name, window_config);
        });
    this->window_manager.RegisterDrawWindowCallback(GUIWindowManager::WindowDrawCallback::PARAM,
        [&, this](const std::string& window_name, GUIWindowManager::WindowConfiguration& window_config) {
            this->drawParametersCallback(window_name, window_config);
        });
    this->window_manager.RegisterDrawWindowCallback(GUIWindowManager::WindowDrawCallback::FPSMS,
        [&, this](const std::string& window_name, GUIWindowManager::WindowConfiguration& window_config) {
            this->drawFpsWindowCallback(window_name, window_config);
        });
    this->window_manager.RegisterDrawWindowCallback(GUIWindowManager::WindowDrawCallback::FONT,
        [&, this](const std::string& window_name, GUIWindowManager::WindowConfiguration& window_config) {
            this->drawFontWindowCallback(window_name, window_config);
        });
    this->window_manager.RegisterDrawWindowCallback(GUIWindowManager::WindowDrawCallback::TF,
        [&, this](const std::string& window_name, GUIWindowManager::WindowConfiguration& window_config) {
            this->drawTFWindowCallback(window_name, window_config);
        });

    // Create window configurations
    GUIWinConfig tmp_win;
    // Main Window ------------------------------------------------------------
    tmp_win.show = true;
    tmp_win.hotkey = core::view::KeyCode(core::view::Key::KEY_F12);
    tmp_win.flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_HorizontalScrollbar;
    tmp_win.callback = GUIWindowManager::WindowDrawCallback::MAIN;
    this->window_manager.AddWindowConfiguration("MegaMol", tmp_win);

    // FPS overlay Window -----------------------------------------------------
    tmp_win.show = false;
    tmp_win.hotkey = core::view::KeyCode(core::view::Key::KEY_F11);
    tmp_win.flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
    tmp_win.callback = GUIWindowManager::WindowDrawCallback::FPSMS;
    this->window_manager.AddWindowConfiguration("FPS", tmp_win);

    // Font Selection Window --------------------------------------------------
    tmp_win.show = false;
    tmp_win.hotkey = core::view::KeyCode(core::view::Key::KEY_F10);
    tmp_win.flags = ImGuiWindowFlags_AlwaysAutoResize;
    tmp_win.callback = GUIWindowManager::WindowDrawCallback::FONT;
    this->window_manager.AddWindowConfiguration("Fonts", tmp_win);

    // Demo Window --------------------------------------------------
    tmp_win.show = false;
    tmp_win.hotkey = core::view::KeyCode(core::view::Key::KEY_F9);
    tmp_win.flags = ImGuiWindowFlags_AlwaysAutoResize;
    tmp_win.callback = GUIWindowManager::WindowDrawCallback::TF;
    this->window_manager.AddWindowConfiguration("Transfer Function Editor", tmp_win);

    // Style settings ---------------------------------------------------------
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.FrameRounding = 3.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.AntiAliasedLines = true;
    style.AntiAliasedFill = true;
    style.DisplayWindowPadding = ImVec2(5.0f, 5.0f);
    style.DisplaySafeAreaPadding = ImVec2(5.0f, 5.0f);
    // Custom style color
    style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.24f, 0.52f, 0.88f, 1.0f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.26f, 0.59f, 0.98f, 0.60f);

    ImGui::SetColorEditOptions(ImGuiColorEditFlags_Uint8 | ImGuiColorEditFlags_RGB |
                               ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_AlphaBar |
                               ImGuiColorEditFlags_AlphaPreview);

    // IO settings ------------------------------------------------------------
    ImGuiIO& io = ImGui::GetIO();
    io.IniSavingRate = 5.0f;  //  in seconds
    io.IniFilename = nullptr; // "imgui.ini";      // disabled, will be replaced by own more comprehensive version
    io.LogFilename = "imgui_log.txt"; // (set to nullptr to disable)
    io.FontAllowUserScaling = true;

    // Adding additional utf-8 glyph ranges
    /// (there is no error if glyph has no representation in font atlas)
    this->font_utf8_ranges.emplace_back(0x0020);
    this->font_utf8_ranges.emplace_back(0x00FF); // Basic Latin + Latin Supplement
    this->font_utf8_ranges.emplace_back(0x20AC);
    this->font_utf8_ranges.emplace_back(0x20AC); // €
    this->font_utf8_ranges.emplace_back(0x2122);
    this->font_utf8_ranges.emplace_back(0x2122); // ™
    this->font_utf8_ranges.emplace_back(0x212B);
    this->font_utf8_ranges.emplace_back(0x212B); // Å
    this->font_utf8_ranges.emplace_back(0x0391);
    this->font_utf8_ranges.emplace_back(0x03D6); // greek alphabet
    this->font_utf8_ranges.emplace_back(0);      // (range termination)

    // Load initial fonts only once for all imgui contexts
    if (!other_context) {
        ImFontConfig config;
        config.OversampleH = 4;
        config.OversampleV = 1;
        config.GlyphRanges = this->font_utf8_ranges.data();
        // Add default font
        io.Fonts->AddFontDefault(&config);
        // Add other known fonts
        std::string font_file, font_path;
        const vislib::Array<vislib::StringW>& searchPaths =
            this->GetCoreInstance()->Configuration().ResourceDirectories();
        for (int i = 0; i < searchPaths.Count(); ++i) {
            font_file = "Proggy_Tiny.ttf";
            font_path = this->SearchFilePathRecursive(font_file, std::wstring(searchPaths[i].PeekBuffer()));
            if (!font_path.empty()) {
                io.Fonts->AddFontFromFileTTF(font_path.c_str(), 10.0f, &config);
            }
            font_file = "Roboto_Regular.ttf";
            font_path = this->SearchFilePathRecursive(font_file, std::wstring(searchPaths[i].PeekBuffer()));
            if (!font_path.empty()) {
                io.Fonts->AddFontFromFileTTF(font_path.c_str(), 18.0f, &config);
            }
            font_file = "Ubuntu_Mono_Regular.ttf";
            font_path = this->SearchFilePathRecursive(font_file, std::wstring(searchPaths[i].PeekBuffer()));
            if (!font_path.empty()) {
                io.Fonts->AddFontFromFileTTF(font_path.c_str(), 15.0f, &config);
            }
            font_file = "Evolventa-Regular.ttf";
            font_path = this->SearchFilePathRecursive(font_file, std::wstring(searchPaths[i].PeekBuffer()));
            if (!font_path.empty()) {
                io.Fonts->AddFontFromFileTTF(font_path.c_str(), 20.0f, &config);
            }
        }
    }

    // ImGui Key Map
    io.KeyMap[ImGuiKey_Tab] = static_cast<int>(core::view::Key::KEY_TAB);
    io.KeyMap[ImGuiKey_LeftArrow] = static_cast<int>(core::view::Key::KEY_LEFT);
    io.KeyMap[ImGuiKey_RightArrow] = static_cast<int>(core::view::Key::KEY_RIGHT);
    io.KeyMap[ImGuiKey_UpArrow] = static_cast<int>(core::view::Key::KEY_UP);
    io.KeyMap[ImGuiKey_DownArrow] = static_cast<int>(core::view::Key::KEY_DOWN);
    io.KeyMap[ImGuiKey_PageUp] = static_cast<int>(core::view::Key::KEY_PAGE_UP);
    io.KeyMap[ImGuiKey_PageDown] = static_cast<int>(core::view::Key::KEY_PAGE_DOWN);
    io.KeyMap[ImGuiKey_Home] = static_cast<int>(core::view::Key::KEY_HOME);
    io.KeyMap[ImGuiKey_End] = static_cast<int>(core::view::Key::KEY_END);
    io.KeyMap[ImGuiKey_Insert] = static_cast<int>(core::view::Key::KEY_INSERT);
    io.KeyMap[ImGuiKey_Delete] = static_cast<int>(core::view::Key::KEY_DELETE);
    io.KeyMap[ImGuiKey_Backspace] = static_cast<int>(core::view::Key::KEY_BACKSPACE);
    io.KeyMap[ImGuiKey_Space] = static_cast<int>(core::view::Key::KEY_SPACE);
    io.KeyMap[ImGuiKey_Enter] = static_cast<int>(core::view::Key::KEY_ENTER);
    io.KeyMap[ImGuiKey_Escape] = static_cast<int>(core::view::Key::KEY_ESCAPE);
    io.KeyMap[ImGuiKey_A] = static_cast<int>(GuiTextModHotkeys::CTRL_A);
    io.KeyMap[ImGuiKey_C] = static_cast<int>(GuiTextModHotkeys::CTRL_C);
    io.KeyMap[ImGuiKey_V] = static_cast<int>(GuiTextModHotkeys::CTRL_V);
    io.KeyMap[ImGuiKey_X] = static_cast<int>(GuiTextModHotkeys::CTRL_X);
    io.KeyMap[ImGuiKey_Y] = static_cast<int>(GuiTextModHotkeys::CTRL_Y);
    io.KeyMap[ImGuiKey_Z] = static_cast<int>(GuiTextModHotkeys::CTRL_Z);

    return true;
} // namespace gui


/**
 * GUIRenderer<M, C>::release
 */
template <class M, class C> void GUIRenderer<M, C>::release() {

    if (this->imgui_context != nullptr) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui::DestroyContext(this->imgui_context);
    }
}


/**
 * GUIRenderer<M, C>::OnKey
 */
template <class M, class C>
bool GUIRenderer<M, C>::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {

    ImGui::SetCurrentContext(this->imgui_context);

    ImGuiIO& io = ImGui::GetIO();

    bool last_return_key = io.KeysDown[static_cast<size_t>(core::view::Key::KEY_ENTER)];
    bool last_num_enter_key = io.KeysDown[static_cast<size_t>(core::view::Key::KEY_KP_ENTER)];

    auto keyIndex = static_cast<size_t>(key);
    switch (action) {
    case core::view::KeyAction::PRESS:
        io.KeysDown[keyIndex] = true;
        break;
    case core::view::KeyAction::RELEASE:
        io.KeysDown[keyIndex] = false;
        break;
    default:
        break;
    }
    io.KeyCtrl = mods.test(core::view::Modifier::CTRL);
    io.KeyShift = mods.test(core::view::Modifier::SHIFT);
    io.KeyAlt = mods.test(core::view::Modifier::ALT);


    // Pass NUM 'Enter' as alternative for 'Return' to ImGui
    bool cur_return_key = ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_ENTER));
    bool cur_num_enter_key = ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_KP_ENTER));
    bool return_pressed = (!last_return_key && cur_return_key);
    bool enter_pressed = (!last_num_enter_key && cur_num_enter_key);
    io.KeysDown[static_cast<size_t>(core::view::Key::KEY_ENTER)] = (return_pressed || enter_pressed);

    // Check for additional text modification hotkeys
    if (action == core::view::KeyAction::RELEASE) {
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_A)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_C)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_V)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_X)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_Y)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_Z)] = false;
    }
    bool hotkeyPressed = true;
    if (io.KeyCtrl && ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_A))) {
        keyIndex = static_cast<size_t>(GuiTextModHotkeys::CTRL_A);
    } else if (io.KeyCtrl && ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_C))) {
        keyIndex = static_cast<size_t>(GuiTextModHotkeys::CTRL_C);
    } else if (io.KeyCtrl && ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_V))) {
        keyIndex = static_cast<size_t>(GuiTextModHotkeys::CTRL_V);
    } else if (io.KeyCtrl && ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_X))) {
        keyIndex = static_cast<size_t>(GuiTextModHotkeys::CTRL_X);
    } else if (io.KeyCtrl && ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_Y))) {
        keyIndex = static_cast<size_t>(GuiTextModHotkeys::CTRL_Y);
    } else if (io.KeyCtrl && ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_Z))) {
        keyIndex = static_cast<size_t>(GuiTextModHotkeys::CTRL_Z);
    } else {
        hotkeyPressed = false;
    }
    if (hotkeyPressed && (action == core::view::KeyAction::PRESS)) {
        io.KeysDown[keyIndex] = true;
    }

    // ------------------------------------------------------------------------
    // NB: Hotkey processing is stopped after first occurence. Order of hotkey processing is crucial.
    // Hotkeys always trigger just oneevent.

    // Exit megamol
    hotkeyPressed = (ImGui::IsKeyDown(io.KeyMap[ImGuiKey_Escape])) ||                               // Escape
                    ((io.KeyAlt) && (ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_F4)))); // Alt + F4
    if (hotkeyPressed) {
        this->shutdown();
        return true;
    }

    // Hotkeys of window(s)
    const auto func = [&, this](const std::string& wn, GUIWinConfig& wc) {
        hotkeyPressed = (ImGui::IsKeyDown(static_cast<int>(wc.hotkey.key))); // Ignoring additional modifiers
        if (hotkeyPressed) {
            if (io.KeyShift) {
                wc.soft_reset = true;
            } else {
                wc.show = !wc.show;
            }
        }
    };
    this->window_manager.EnumWindows(func);

    // Always consume keyboard input if requested by any imgui widget (e.g. text input).
    // User expects hotkey priority of text input thus needs to be processed before parameter hotkeys.
    if (io.WantCaptureKeyboard) {
        return true;
    }

    // Check only considered modules for pressed parameter hotkeys
    std::vector<std::string> modules_list;
    const auto modfunc = [&, this](const std::string& wn, GUIWinConfig& wc) {
        for (auto& m : wc.param_config.modules_list) {
            modules_list.emplace_back(m);
        }
    };
    this->window_manager.EnumWindows(modfunc);
    hotkeyPressed = false;
    const core::Module* current_mod = nullptr;
    bool consider_module = false;
    this->GetCoreInstance()->EnumParameters([&, this](const auto& mod, auto& slot) {
        if (current_mod != &mod) {
            current_mod = &mod;
            consider_module = this->considerModule(mod.FullName().PeekBuffer(), modules_list);
        }

        if (consider_module) {
            auto param = slot.Parameter();
            if (!param.IsNull()) {
                if (auto* p = slot.template Param<core::param::ButtonParam>()) {
                    auto keyCode = p->GetKeyCode();

                    if (hotkeyPressed) return;

                    hotkeyPressed = (ImGui::IsKeyDown(static_cast<int>(keyCode.key))) &&
                                    (keyCode.mods.test(core::view::Modifier::ALT) == io.KeyAlt) &&
                                    (keyCode.mods.test(core::view::Modifier::CTRL) == io.KeyCtrl) &&
                                    (keyCode.mods.test(core::view::Modifier::SHIFT) == io.KeyShift);
                    if (hotkeyPressed) {
                        p->setDirty();
                    }
                }
            }
        }
    });
    if (hotkeyPressed) return true;

    // ------------------------------------------------------------------------

    auto* cr = this->decorated_renderer_slot.template CallAs<C>();
    if (cr == nullptr) return false;

    core::view::InputEvent evt;
    evt.tag = core::view::InputEvent::Tag::Key;
    evt.keyData.key = key;
    evt.keyData.action = action;
    evt.keyData.mods = mods;
    cr->SetInputEvent(evt);
    if (!(*cr)(core::view::InputCall::FnOnKey)) return false;

    return false;
}


/**
 * GUIRenderer<M, C>::OnChar
 */
template <class M, class C> bool GUIRenderer<M, C>::OnChar(unsigned int codePoint) {

    ImGui::SetCurrentContext(this->imgui_context);

    ImGuiIO& io = ImGui::GetIO();
    io.ClearInputCharacters();
    if (codePoint > 0 && codePoint < 0x10000) io.AddInputCharacter((unsigned short)codePoint);

    auto* cr = this->decorated_renderer_slot.template CallAs<C>();
    if (cr) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::Char;
        evt.charData.codePoint = codePoint;
        cr->SetInputEvent(evt);
        if ((*cr)(core::view::InputCall::FnOnChar)) return true;
    }

    return true;
}


/**
 * GUIRenderer<M, C>::OnMouseMove
 */
template <class M, class C> bool GUIRenderer<M, C>::OnMouseMove(double x, double y) {

    ImGui::SetCurrentContext(this->imgui_context);

    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);

    auto hoverFlags = ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenDisabled |
                      ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem;
    if (!ImGui::IsWindowHovered(hoverFlags)) {
        auto* cr = this->decorated_renderer_slot.template CallAs<C>();
        if (cr == nullptr) return false;

        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;
        cr->SetInputEvent(evt);
        if (!(*cr)(core::view::InputCall::FnOnMouseMove)) return false;
    }

    return true;
}


/**
 * GUIRenderer<M, C>::OnMouseButton
 */
template <class M, class C>
bool GUIRenderer<M, C>::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    ImGui::SetCurrentContext(this->imgui_context);

    bool down = (action == core::view::MouseButtonAction::PRESS);
    auto buttonIndex = static_cast<size_t>(button);
    ImGuiIO& io = ImGui::GetIO();
    io.MouseDown[buttonIndex] = down;

    auto hoverFlags = ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenDisabled |
                      ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem;
    if (!ImGui::IsWindowHovered(hoverFlags)) {
        auto* cr = this->decorated_renderer_slot.template CallAs<C>();
        if (cr == nullptr) return false;

        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        cr->SetInputEvent(evt);
        if (!(*cr)(core::view::InputCall::FnOnMouseButton)) return false;
    }

    return (down); // Don't consume 'release' events.
}


/**
 * GUIRenderer<M, C>::OnMouseScroll
 */
template <class M, class C> bool GUIRenderer<M, C>::OnMouseScroll(double dx, double dy) {

    ImGui::SetCurrentContext(this->imgui_context);

    ImGuiIO& io = ImGui::GetIO();
    io.MouseWheelH += (float)dx;
    io.MouseWheel += (float)dy;

    auto hoverFlags =
        ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenDisabled | ImGuiHoveredFlags_AllowWhenBlockedByPopup;
    if (!ImGui::IsWindowHovered(hoverFlags)) {
        auto* cr = this->decorated_renderer_slot.template CallAs<C>();
        if (cr == nullptr) return false;

        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;
        cr->SetInputEvent(evt);
        if (!(*cr)(core::view::InputCall::FnOnMouseScroll)) return false;
    }

    return true;
}


/**
 * GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::GetExtents
 */
template <>
inline bool GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::GetExtents(
    core::view::CallRender2D& call) {

    auto* cr = this->decorated_renderer_slot.CallAs<core::view::CallRender2D>();
    if (cr != nullptr) {
        (*cr) = call;
        if ((*cr)(core::view::AbstractCallRender::FnGetExtents)) {
            call = (*cr);
        }
    } else {
        call.SetBoundingBox(vislib::math::Rectangle<float>(0, 1, 1, 0));
    }

    return true;
}


/**
 * GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::GetExtents
 */
template <>
inline bool GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::GetExtents(
    core::view::CallRender3D& call) {

    auto* cr = this->decorated_renderer_slot.CallAs<core::view::CallRender3D>();
    if (cr != nullptr) {
        (*cr) = call;
        if ((*cr)(core::view::AbstractCallRender::FnGetExtents)) {
            call = (*cr);
        }
    } else {
        call.AccessBoundingBoxes().Clear();
        call.AccessBoundingBoxes().SetWorldSpaceBBox(
            vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));
    }

    return true;
}

/**
 * GUIRenderer<M, C>::Render
 */
template <class M, class C> bool GUIRenderer<M, C>::Render(C& call) {

    if (this->overlay_slot.GetStatus() == core::AbstractSlot::SlotStatus::STATUS_CONNECTED) {
        vislib::sys::Log::DefaultLog.WriteError("[GUIRenderer][Render] Only one connected callee slot is allowed!");
        return false;
    }

    auto* cr = this->decorated_renderer_slot.template CallAs<C>();
    if (cr != nullptr) {
        (*cr) = call;
        if ((*cr)(core::view::AbstractCallRender::FnRender)) {
            call = (*cr);
        }
    }

    return this->render(call.GetViewport(), call.InstanceTime());
}


/**
 * GUIRenderer<M, C>::OnRenderCallback
 */
template <class M, class C> bool GUIRenderer<M, C>::OnOverlayCallback(core::Call& call) {

    if (this->renderSlot.GetStatus() == core::AbstractSlot::SlotStatus::STATUS_CONNECTED) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[GUIRenderer][OnOverlayCallback] Only one connected callee slot is allowed!");
        return false;
    }

    auto* cr = this->decorated_renderer_slot.template CallAs<C>();
    if (cr != nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "[GUIRenderer][OnOverlayCallback] Render callback of connected Renderer is not called!");
    }

    try {
        core::view::CallSplitViewOverlay& cgr = dynamic_cast<core::view::CallSplitViewOverlay&>(call);
        return this->render(cgr.GetViewport(), cgr.InstanceTime());
    } catch (...) {
        ASSERT("OnRenderCallback call cast failed\n");
    }
    return false;
}


/**
 * GUIRenderer<M, C>::OnKeyCallback
 */
template <class M, class C> bool GUIRenderer<M, C>::OnKeyCallback(core::Call& call) {
    try {
        core::view::CallSplitViewOverlay& cr = dynamic_cast<core::view::CallSplitViewOverlay&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == core::view::InputEvent::Tag::Key && "Callback invocation mismatched input event");
        return this->OnKey(evt.keyData.key, evt.keyData.action, evt.keyData.mods);
    } catch (...) {
        ASSERT("OnKeyCallback call cast failed\n");
    }
    return false;
}


/**
 * GUIRenderer<M, C>::OnCharCallback
 */
template <class M, class C> bool GUIRenderer<M, C>::OnCharCallback(core::Call& call) {
    try {
        core::view::CallSplitViewOverlay& cr = dynamic_cast<core::view::CallSplitViewOverlay&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == core::view::InputEvent::Tag::Char && "Callback invocation mismatched input event");
        return this->OnChar(evt.charData.codePoint);
    } catch (...) {
        ASSERT("OnCharCallback call cast failed\n");
    }
    return false;
}


/**
 * GUIRenderer<M, C>::OnMouseButtonCallback
 */
template <class M, class C> bool GUIRenderer<M, C>::OnMouseButtonCallback(core::Call& call) {
    try {
        core::view::CallSplitViewOverlay& cr = dynamic_cast<core::view::CallSplitViewOverlay&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == core::view::InputEvent::Tag::MouseButton && "Callback invocation mismatched input event");
        return this->OnMouseButton(evt.mouseButtonData.button, evt.mouseButtonData.action, evt.mouseButtonData.mods);
    } catch (...) {
        ASSERT("OnMouseButtonCallback call cast failed\n");
    }
    return false;
}


/**
 * GUIRenderer<M, C>::OnMouseMoveCallback
 */
template <class M, class C> bool GUIRenderer<M, C>::OnMouseMoveCallback(core::Call& call) {
    try {
        core::view::CallSplitViewOverlay& cr = dynamic_cast<core::view::CallSplitViewOverlay&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == core::view::InputEvent::Tag::MouseMove && "Callback invocation mismatched input event");
        return this->OnMouseMove(evt.mouseMoveData.x, evt.mouseMoveData.y);
    } catch (...) {
        ASSERT("OnMouseMoveCallback call cast failed\n");
    }
    return false;
}


/**
 * GUIRenderer<M, C>::OnMouseScrollCallback
 */
template <class M, class C> bool GUIRenderer<M, C>::OnMouseScrollCallback(core::Call& call) {
    try {
        core::view::CallSplitViewOverlay& cr = dynamic_cast<core::view::CallSplitViewOverlay&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == core::view::InputEvent::Tag::MouseScroll && "Callback invocation mismatched input event");
        return this->OnMouseScroll(evt.mouseScrollData.dx, evt.mouseScrollData.dy);
    } catch (...) {
        ASSERT("OnMouseScrollCallback call cast failed\n");
    }
    return false;
}


/**
 * GUIRenderer<M, C>::render
 */
template <class M, class C> bool GUIRenderer<M, C>::render(vislib::math::Rectangle<int> viewport, double instanceTime) {

    ImGui::SetCurrentContext(this->imgui_context);

    auto viewportWidth = viewport.Width();
    auto viewportHeight = viewport.Height();

    // Set IO stuff
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)viewportWidth, (float)viewportHeight);
    io.DisplayFramebufferScale = ImVec2(1.0, 1.0);

    if ((instanceTime - this->last_instance_time) < 0.0) {
        vislib::sys::Log::DefaultLog.WriteWarn("[GUIRenderer] Current instance time results in negative time delta.");
    }
    io.DeltaTime = ((instanceTime - this->last_instance_time) > 0.0)
                       ? (static_cast<float>(instanceTime - this->last_instance_time))
                       : (io.DeltaTime);
    this->last_instance_time =
        ((instanceTime - this->last_instance_time) > 0.0) ? (instanceTime) : (this->last_instance_time + io.DeltaTime);

    // Loading new font (before next ImGui::Begin!)
    if (!this->load_new_font_filename.empty()) {
        ImFontConfig config;
        config.OversampleH = 4;
        config.OversampleV = 1;
        config.GlyphRanges = this->font_utf8_ranges.data();
        io.Fonts->AddFontFromFileTTF(this->load_new_font_filename.c_str(), this->load_new_font_size, &config);
        ImGui_ImplOpenGL3_CreateFontsTexture();
        // Load last added font
        io.FontDefault = io.Fonts->Fonts[(io.Fonts->Fonts.Size - 1)];
        this->load_new_font_filename.clear();
    }

    // Applying new window configuration profile (before next ImGui::Begin!)
    if (!this->load_new_profile.empty()) {
        this->window_manager.LoadWindowConfigurationProfile(this->load_new_profile);
        this->load_new_profile.clear();
    }

    // Start new frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();

    // Draw windows
    const auto func = [&, this](const std::string& wn, GUIWinConfig& wc) {
        if (wc.show) {
            ImGui::SetNextWindowBgAlpha(1.0f);

            if (!ImGui::Begin(wn.c_str(), &wc.show, wc.flags)) {
                ImGui::End(); // early ending
                return;
            }

            // Apply soft reset of window position and size
            if (wc.soft_reset) {
                this->window_manager.SoftResetWindowSizePos(wn, wc);
                wc.soft_reset = false;
            }
            // Apply reset after new profile has been loaded
            if (wc.reset) {
                this->window_manager.ResetWindowOnProfileLoad(wn, wc);
                wc.reset = false;
            }

            this->window_manager.DrawWindowContent(wn);

            // Saving current window position and size to all window configurations for possible profile saving.
            wc.position = ImGui::GetWindowPos();
            wc.size = ImGui::GetWindowSize();

            ImGui::End();
        }
    };
    this->window_manager.EnumWindows(func);

    // Render the frame
    glViewport(0, 0, viewportWidth, viewportHeight);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return true;
}


/**
 * GUIRenderer<M, C>::drawMainWindowCallback
 */
template <class M, class C>
void GUIRenderer<M, C>::drawMainWindowCallback(const std::string& window_name, GUIWinConfig& window_config) {

    // Menu -------------------------------------------------------------------
    /// Requires window flag ImGuiWindowFlags_MenuBar
    if (ImGui::BeginMenuBar()) {
        this->drawMenu();
        ImGui::EndMenuBar();
    }

    // Parameters -------------------------------------------------------------
    ImGui::Text("Parameters");
    std::string color_param_help = "[Hover] Parameter for Description Tooltip\n[Right-Click] for Context Menu\n[Drag & "
                                   "Drop] Module Header to other Parameter Window";
    this->HelpMarkerToolTip(color_param_help);

    this->drawParametersCallback(window_name, window_config);
}


/**
 * GUIRenderer<M, C>::drawTFWindowCallback
 */
template <class M, class C>
void GUIRenderer<M, C>::drawTFWindowCallback(const std::string& window_name, GUIWinConfig& window_config) {

    this->tf_editor.DrawTransferFunctionEditor();
}


/**
 * GUIRenderer<M, C>::drawParametersCallback
 */
template <class M, class C>
void GUIRenderer<M, C>::drawParametersCallback(const std::string& window_name, GUIWinConfig& window_config) {

    ImGuiStyle& style = ImGui::GetStyle();

    GUIWindowManager::ParamConfig* pc = &window_config.param_config;

    ImGui::PushItemWidth(ImGui::GetContentRegionAvailWidth() * 0.5f); // set general proportional item width

    // Options
    int overrideState = -1; /// invalid
    if (ImGui::Button("Expand All")) {
        overrideState = 1; /// open
    }
    ImGui::SameLine();
    if (ImGui::Button("Collapse All")) {
        overrideState = 0; /// close
    }

    bool show_only_hotkeys = pc->show_hotkeys;
    ImGui::Checkbox("Show Hotkeys", &show_only_hotkeys);
    pc->show_hotkeys = show_only_hotkeys;

    std::map<int, std::string> opts;
    opts[(int)GUIWindowManager::FilterMode::ALL] = "All";
    opts[(int)GUIWindowManager::FilterMode::INSTANCE] = "Instance";
    opts[(int)GUIWindowManager::FilterMode::VIEW] = "View";
    unsigned int opts_cnt = (unsigned int)opts.size();
    if (ImGui::BeginCombo("Module Filter", opts[(int)pc->module_filter].c_str())) {
        for (unsigned int i = 0; i < opts_cnt; ++i) {

            if (ImGui::Selectable(opts[i].c_str(), ((int)pc->module_filter == i))) {
                pc->module_filter = (GUIWindowManager::FilterMode)i;
                pc->modules_list.clear();
                if ((pc->module_filter == GUIWindowManager::FilterMode::INSTANCE) ||
                    (pc->module_filter == GUIWindowManager::FilterMode::VIEW)) {

                    // Goal is to find view module with shortest call connection path to this gui rendere module.
                    // Since enumeration of modules goes bottom up, result for first abstract view is stored and
                    // following hits are ignored.
                    std::string viewname;
                    std::string thisname = this->FullName().PeekBuffer();
                    const auto view_func = [&, this](core::Module* viewmod) {
                        auto v = dynamic_cast<core::view::AbstractView*>(viewmod);
                        if (v != nullptr) {
                            std::string vname = v->FullName().PeekBuffer();

                            bool found = false;
                            const auto find_func = [&, this](core::Module* guimod) {
                                std::string modname = guimod->FullName().PeekBuffer();
                                if (thisname == modname) {
                                    found = true;
                                }
                            };
                            this->GetCoreInstance()->EnumModulesNoLock(viewmod, find_func);

                            if (found && viewname.empty()) {
                                viewname = vname;
                            }
                        }
                    };
                    this->GetCoreInstance()->EnumModulesNoLock(nullptr, view_func);

                    if (!viewname.empty()) {
                        if (pc->module_filter == GUIWindowManager::FilterMode::INSTANCE) {
                            // Considering modules depending on the INSTANCE NAME of the first view this gui
                            // renderer is connected to.
                            std::string instname = "";
                            if (viewname.find("::", 2) != std::string::npos) {
                                instname = viewname.substr(0, viewname.find("::", 2));
                            }
                            if (!instname.empty()) { /// Consider all modules if view is not assigned to any instance
                                const auto func = [&, this](core::Module* mod) {
                                    std::string modname = mod->FullName().PeekBuffer();
                                    bool foundInstanceName = (modname.find(instname) != std::string::npos);
                                    // Modules with no namespace are always taken into account ...
                                    bool noInstanceNamePresent = (modname.find("::", 2) == std::string::npos);
                                    if (foundInstanceName || noInstanceNamePresent) {
                                        pc->modules_list.emplace_back(modname);
                                    }
                                };
                                this->GetCoreInstance()->EnumModulesNoLock(nullptr, func);
                            }
                        } else { // (pc->module_filter == GUIWindowManager::FilterMode::VIEW)
                            // Considering modules depending on their connection to the first VIEW this gui renderer
                            // is connected to.
                            const auto add_func = [&, this](core::Module* mod) {
                                std::string modname = mod->FullName().PeekBuffer();
                                pc->modules_list.emplace_back(modname);
                            };
                            this->GetCoreInstance()->EnumModulesNoLock(viewname, add_func);
                        }
                    } else {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "[GUIRenderer] Couldn't find abstract view module this gui renderer is connected to.");
                    }
                }
            }
            std::string hover = "Show all Modules."; // == GUIWindowManager::FilterMode::ALL
            if (i == (int)GUIWindowManager::FilterMode::INSTANCE) {
                hover = "Show Modules with same Instance Name as current View and Modules with no Instance Name.";
            } else if (i == (int)GUIWindowManager::FilterMode::VIEW) {
                hover = "Show Modules subsequently connected to the View Module the Gui Module is connected to.";
            }
            this->HoverToolTip(hover);
        }
        ImGui::EndCombo();
    }
    this->HelpMarkerToolTip("Filter applies globally to all parameter windows.\nSelected filter is not refreshed on "
                            "graph changes.\nSelect filter again to trigger refresh.");
    ImGui::Separator();

    // Listing parameters
    const core::Module* current_mod = nullptr;
    bool current_mod_open = false;
    size_t dnd_size = GUI_MAX_BUFFER_LEN; // Set same max size of all module labels for drag and drop.
    std::string param_namespace = "";
    unsigned int param_indent_stack = 0;
    bool param_namespace_open = true;

    this->GetCoreInstance()->EnumParameters([&, this](const auto& mod, auto& slot) {
        // Check for new module
        if (current_mod != &mod) {
            current_mod = &mod;
            std::string label = mod.FullName().PeekBuffer();

            // Reset parameter namespace stuff
            param_namespace = "";
            param_namespace_open = true;
            while (param_indent_stack > 0) {
                param_indent_stack--;
                ImGui::Unindent();
            }

            // Check if module should be considered.
            if (!this->considerModule(label, pc->modules_list)) {
                current_mod_open = false;
                return;
            }

            // Main parameter window always draws all module's parameters
            if (window_config.callback != GUIWindowManager::WindowDrawCallback::MAIN) {
                // Consider only modules contained in list
                if (std::find(pc->modules_list.begin(), pc->modules_list.end(), label) == pc->modules_list.end()) {
                    current_mod_open = false;
                    return;
                }
            }

            auto headerId = ImGui::GetID(label.c_str());
            auto headerState = overrideState;
            if (headerState == -1) {
                headerState = ImGui::GetStateStorage()->GetInt(headerId, 0); // 0=close 1=open
            }

            ImGui::GetStateStorage()->SetInt(headerId, headerState);
            current_mod_open = ImGui::CollapsingHeader(label.c_str(), nullptr);

            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Copy to new Window")) {
                    std::string window_name =
                        "Parameters###parameters" +
                        std::to_string(this->last_instance_time); /// using instance time as hidden unique id
                    GUIWinConfig tmp_win;
                    tmp_win.show = true;
                    tmp_win.flags = ImGuiWindowFlags_HorizontalScrollbar;
                    tmp_win.callback = GUIWindowManager::WindowDrawCallback::PARAM;
                    tmp_win.param_config.show_hotkeys = false;
                    tmp_win.param_config.modules_list.emplace_back(label);
                    this->window_manager.AddWindowConfiguration(window_name, tmp_win);
                }
                // Deleting module's parameters is not available in main parameter window.
                if (window_config.callback !=
                    GUIWindowManager::WindowDrawCallback::MAIN) { // && (pc->modules_list.size() > 1)) {
                    if (ImGui::MenuItem("Delete from List")) {
                        std::vector<std::string>::iterator find_iter =
                            std::find(pc->modules_list.begin(), pc->modules_list.end(), label);
                        // Break if module name is not contained in list
                        if (find_iter != pc->modules_list.end()) {
                            pc->modules_list.erase(find_iter);
                        }
                    }
                }
                ImGui::EndPopup();
            }

            // Drag source
            label.resize(dnd_size);
            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
                ImGui::SetDragDropPayload("DND_COPY_MODULE_PARAMETERS", label.c_str(), (label.size() * sizeof(char)));
                ImGui::Text(label.c_str());
                ImGui::EndDragDropSource();
            }
        }

        if (current_mod_open) {
            // Check for new parameter namespace
            std::string param_name = slot.Name().PeekBuffer();
            auto pos = param_name.find("::");
            std::string current_param_namespace = "";
            if (pos != std::string::npos) {
                current_param_namespace = param_name.substr(0, pos);
            }
            if (current_param_namespace != param_namespace) {

                param_namespace = current_param_namespace;

                while (param_indent_stack > 0) {
                    param_indent_stack--;
                    ImGui::Unindent();
                }

                ImGui::Separator();
                if (!param_namespace.empty()) {
                    ImGui::Indent();
                    std::string label = param_namespace + "###" + param_namespace + "__" + param_name;
                    param_namespace_open = ImGui::CollapsingHeader(label.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
                    param_indent_stack++;
                } else {
                    param_namespace_open = true;
                }
            }

            // Draw parameter
            if (param_namespace_open) {
                if (pc->show_hotkeys) {
                    this->drawParameterHotkey(mod, slot);
                } else {
                    this->drawParameter(mod, slot);
                }
            }
        }
    });
    // Reset parameter namespace stuff
    while (param_indent_stack > 0) {
        param_indent_stack--;
        ImGui::Unindent();
    }

    // Drop target
    ImGui::InvisibleButton("Drop Area", ImVec2(ImGui::GetContentRegionAvailWidth(), ImGui::GetFontSize()));
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_COPY_MODULE_PARAMETERS")) {

            IM_ASSERT(payload->DataSize == (dnd_size * sizeof(char)));
            std::string payload_id = (const char*)payload->Data;

            // Nothing to add to main parameter window (draws always all module's parameters)
            if ((window_config.callback != GUIWindowManager::WindowDrawCallback::MAIN)) {
                // Insert dragged module name only if not contained in list
                if (std::find(pc->modules_list.begin(), pc->modules_list.end(), payload_id) == pc->modules_list.end()) {
                    pc->modules_list.emplace_back(payload_id);
                }
            }
        }
        ImGui::EndDragDropTarget();
    }

    ImGui::PopItemWidth();
}


/**
 * GUIRenderer<M, C>::drawFpsWindowCallback
 */
template <class M, class C>
void GUIRenderer<M, C>::drawFpsWindowCallback(const std::string& window_name, GUIWinConfig& window_config) {

    // Updateing values
    ImGuiIO& io = ImGui::GetIO();

    GUIWindowManager::FpsMsConfig* fmc = &window_config.fpsms_config;

    fmc->current_delay += io.DeltaTime;

    const float max_delay = fmc->max_delay;
    const float max_value_count = fmc->max_value_count;

    /*
    if (max_delay <= 0.0f) {
        return;
    }
    if (max_value_count == 0) {
        this->fps_values.clear();
        this->ms_values.clear();
        return;
    }

    if (c.current_delay > (1.0f / max_delay)) {

        // Leave some space in histogram for text of current value
        const float scale_fac = 1.5f;

        if (this->fps_values.size() != this->ms_values.size()) {
            vislib::sys::Log::DefaultLog.WriteError(
                "[GUIRenderer][updateFps] Fps and ms value arrays don't have same size.");
            return;
        }

        size_t size = this->fps_values.size();
        if (size != max_value_count) {
            if (size > max_value_count) {
                this->fps_values.erase(this->fps_values.begin(), this->fps_values.begin() + (size - max_value_count));
                this->ms_values.erase(this->ms_values.begin(), this->ms_values.begin() + (size - max_value_count));

            } else if (size < max_value_count) {
                this->fps_values.insert(this->fps_values.begin(), (max_value_count - size), 0.0f);
                this->ms_values.insert(this->ms_values.begin(), (max_value_count - size), 0.0f);
            }
        }
        if (size > 0) {
            this->fps_values.erase(this->fps_values.begin());
            this->ms_values.erase(this->ms_values.begin());

            this->fps_values.emplace_back(io.Framerate);
            this->ms_values.emplace_back(io.DeltaTime * 1000.0f); // scale to milliseconds

            float value_max = 0.0f;
            for (auto& v : this->fps_values) {
                value_max = (v > value_max) ? (v) : (value_max);
            }
            this->fps_value_scale = value_max * scale_fac;

            value_max = 0.0f;
            for (auto& v : this->ms_values) {
                value_max = (v > value_max) ? (v) : (value_max);
            }
            this->ms_value_scale = value_max * scale_fac;
        }

        fpsms_currc.current_delayent_delay = 0.0f;
    }

    // Draw window content
    if (ImGui::RadioButton("fps", (window_config.fpsms_mode == GUIWindowManager::FpsMsMode::FPS))) {
        window_config.fpsms_mode = GUIWindowManager::FpsMsMode::FPS;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("ms", (window_config.fpsms_mode == GUIWindowManager::FpsMsMode::MS))) {
        window_config.fpsms_mode = GUIWindowManager::FpsMsMode::MS;
    }

    ImGui::SameLine(0.0f, 50.0f);
    ImGui::Checkbox("Options", &this->fpsms_show_options);

    // Default for window_config.fpsms_mode == GUIWindowManager::FpsMsMode::FPS
    std::vector<float>* arr = &this->fps_values;
    float val_scale = this->fps_value_scale;
    if (window_config.fpsms_mode == GUIWindowManager::FpsMsMode::MS) {
        arr = &this->ms_values;
        val_scale = this->ms_value_scale;
    }
    float* data = arr->data();
    int count = (int)arr->size();

    std::string val;
    if (!arr->empty()) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(3) << arr->back();
        val = stream.str();
    }
    ImGui::PlotHistogram(
        "###fpsmsplot", data, count, 0, val.c_str(), 0.0f, val_scale, ImVec2(0.0f, 50.0f)); /// use hidden label

    if (this->fpsms_show_options) {

        if (ImGui::InputFloat(
                "Refresh Rate", &this->fpsms_max_delay, 0.0f, 0.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            this->fps_values.clear();
            this->ms_values.clear();
        }
        // Validate refresh rate
        this->fpsms_max_delay = std::max(0.0f, this->fpsms_max_delay);
        std::string help = "Changes clear all values";
        this->HelpMarkerToolTip(help);

        int mvc = (int)max_value_count;
        ImGui::InputInt("Stored Values Count", &mvc, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue);
        // Validate refresh rate
        max_value_count = (size_t)(std::max(0, mvc));

        if (ImGui::Button("Current Value")) {
            ImGui::SetClipboardText(val.c_str());
        }
        ImGui::SameLine();

        if (ImGui::Button("All Values")) {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(3);
            auto end = (*arr).rend();
            for (std::vector<float>::reverse_iterator i = (*arr).rbegin(); i != end; ++i) {
                stream << (*i) << "\n";
            }
            ImGui::SetClipboardText(stream.str().c_str());
        }
        ImGui::SameLine();

        ImGui::Text("Copy to Clipborad");
        help = "Values are copied in chronological order (newest first)";
        this->HelpMarkerToolTip(help);
    }
    */
}


/**
 * GUIRenderer<M, C>::drawFontWindowCallback
 */
template <class M, class C>
void GUIRenderer<M, C>::drawFontWindowCallback(const std::string& window_name, GUIWinConfig& window_config) {

    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    ImFont* font_current = ImGui::GetFont();

    GUIWindowManager::FontConfig* fc = &window_config.font_config;

    /*
    if (ImGui::BeginCombo("Select available Font", font_current->GetDebugName())) {
        for (int n = 0; n < io.Fonts->Fonts.Size; n++) {
            if (ImGui::Selectable(io.Fonts->Fonts[n]->GetDebugName(), (io.Fonts->Fonts[n] == font_current)))
                io.FontDefault = io.Fonts->Fonts[n];
        }
        ImGui::EndCombo();
    }
    // Saving current font to window configuration.
    window_config.font_name = std::string(font_current->GetDebugName());

    ImGui::Separator();
    ImGui::Text("Load new Font from File");

    std::string label = "Font Filename (.ttf)";
    vislib::StringA valueString;
    vislib::UTF8Encoder::Encode(valueString, vislib::StringA(this->load_new_font_filename.c_str()));
    size_t bufferLength = GUI_MAX_BUFFER_LEN;
    char* buffer = new char[bufferLength];
    memcpy(buffer, valueString.PeekBuffer(), valueString.Length() + 1);
    ImGui::InputText(label.c_str(), buffer, bufferLength);
    vislib::UTF8Encoder::Decode(valueString, vislib::StringA(buffer));
    this->load_new_font_filename = valueString.PeekBuffer();
    delete[] buffer;

    label = "Font Size";
    ImGui::InputFloat(label.c_str(), &this->load_new_font_size, 0.0f, 0.0f, "%.2f", ImGuiInputTextFlags_None);
    // Validate font size
    if (this->load_new_font_size <= 0.0f) {
        this->load_new_font_size = 5.0f; /// min valid font size
    }

    // Validate font file before offering load button
    if (this->FileHasExtension(this->load_new_font_filename, std::string(".ttf"))) {
        if (ImGui::Button("Add Font")) {
            this->font_new_load = true;
        }
    } else {
        ImGui::TextColored(style.Colors[ImGuiCol_ButtonHovered], "Please enter valid font file name");
    }
    std::string help = "Same font can be loaded multiple times using different font size";
    this->HelpMarkerToolTip(help);
    */
}


/**
 * GUIRenderer<M, C>::drawMenu
 */
template <class M, class C> void GUIRenderer<M, C>::drawMenu(void) {

    // App
    bool save_profile = false;
    if (ImGui::BeginMenu("App")) {

        //    // Load/save parameter values to LUA file
        //    if (ImGui::Button("Save Parameters to File")) {
        //
        //        // Save parameter file
        //    }
        //    ImGui::SameLine();
        //    if (ImGui::Button("Load Parameters from File")) {
        //        if (!this->FilePathExists(this->param_file)) {
        //            ImGui::TextColored(style.Colors[ImGuiCol_ButtonHovered], "Please enter valid Paramter File
        //            Name");
        //        } else
        //        {
        //            // Load parameter file
        //        }
        //    }
        //    ... popup asking for file name ...

        // Exit program
        if (ImGui::MenuItem("Exit", "'Esc', ALT + 'F4'")) {
            this->shutdown();
        }
        ImGui::EndMenu();
    }

    // Windows
    if (ImGui::BeginMenu("View")) {
        const auto func = [&, this](const std::string& wn, GUIWinConfig& wc) {
            bool win_open = wc.show;
            std::string hotkey_label = wc.hotkey.ToString();
            if (!hotkey_label.empty()) {
                hotkey_label = "(SHIFT +) " + hotkey_label;
            }
            if (ImGui::MenuItem(wn.c_str(), hotkey_label.c_str(), &win_open)) {
                wc.show = !wc.show;
            }
            this->HoverToolTip("[Shift] + ['Window Hotkey'] to Reset Size and Position of Window");
        };
        this->window_manager.EnumWindows(func);
        ImGui::Separator();

        // GUI window Profile
        if (ImGui::BeginMenu("Window Configuration")) {
            if (ImGui::BeginMenu("Load Profile")) {
                std::list<std::string> profile_list = this->window_manager.GetWindowConfigurationProfileList();
                for (auto& p : profile_list) {
                    if (ImGui::MenuItem(p.c_str())) {
                        // Remember profile name for delayed loading
                        this->load_new_profile = p;
                    }
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Delete Profile")) {
                std::list<std::string> profile_list = this->window_manager.GetWindowConfigurationProfileList();
                for (auto& p : profile_list) {
                    if (ImGui::MenuItem(p.c_str())) {
                        this->window_manager.DeleteWindowConfigurationProfile(p);
                    }
                }
                ImGui::EndMenu();
            }
            if (ImGui::MenuItem("Save Profile")) {
                save_profile = true; /// Processed below because of popup ...
            }
            ImGui::EndMenu();
        }

        ImGui::EndMenu();
    }

    // Help
    bool open_popup = false;
    if (ImGui::BeginMenu("Help")) {
        const std::string gitLink = "https://github.com/UniStuttgart-VISUS/megamol";
        const std::string mmLink = "https://megamol.org/";
        const std::string helpLink = "https://github.com/UniStuttgart-VISUS/megamol/blob/master/Readme.md";
        const std::string hint = "Copy Link to Clipboard";
        if (ImGui::MenuItem("GitHub")) {
            ImGui::SetClipboardText(gitLink.c_str());
        }
        this->HoverToolTip(hint);
        if (ImGui::MenuItem("Readme")) {
            ImGui::SetClipboardText(helpLink.c_str());
        }
        this->HoverToolTip(hint);
        if (ImGui::MenuItem("Web Page")) {
            ImGui::SetClipboardText(mmLink.c_str());
        }
        this->HoverToolTip(hint);
        ImGui::Separator();
        if (ImGui::MenuItem("About...")) {
            open_popup = true;
        }
        ImGui::EndMenu();
    }

    // PopUps
    std::string about =
        std::string("MegaMol is GREAT!\nUsing Dear ImGui ") + std::string(IMGUI_VERSION) + std::string("\n");
    if (open_popup) {
        ImGui::OpenPopup("About");
    }
    if (ImGui::BeginPopupModal("About", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text(about.c_str());
        ImGui::Separator();
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::SetItemDefaultFocus();
        ImGui::EndPopup();
    }

    // Process request for saving current window profile
    std::string profile_name =
        this->InputDialogPopUp("Save Current Window Configuration", "Profile Name", save_profile);
    if (!profile_name.empty()) {
        this->window_manager.SaveWindowConfigurationProfile(profile_name);
    }
}


/**
 * GUIRenderer<M, C>::drawParameter
 */
template <class M, class C>
void GUIRenderer<M, C>::drawParameter(const core::Module& mod, core::param::ParamSlot& slot) {

    ImGuiStyle& style = ImGui::GetStyle();
    std::string help;

    auto param = slot.Parameter();
    if (!param.IsNull() && param->IsGUIVisible()) {
        // Set different style if parameter is read-only
        if (param->IsGUIReadOnly()) {
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.25f);
        }

        std::string param_name = slot.Name().PeekBuffer();
        std::string param_id = std::string(mod.FullName().PeekBuffer()) + "::" + param_name;
        auto pos = param_name.find("::");
        if (pos != std::string::npos) {
            param_name = param_name.substr(pos + 2);
        }
        std::string param_label = param_name + "###" + param_id;
        std::string param_desc = slot.Description().PeekBuffer();
        std::string float_format = "%.7f";

        if (auto* p = slot.template Param<core::param::BoolParam>()) {
            auto value = p->Value();
            if (ImGui::Checkbox(param_label.c_str(), &value)) {
                p->SetValue(value);
            }
        } else if (auto* p = slot.template Param<core::param::ButtonParam>()) {
            std::string hotkey = " (";
            hotkey.append(p->GetKeyCode().ToString());
            hotkey.append(")");
            auto insert_pos = param_label.find("###"); // no check if found -> should be present
            param_label.insert(insert_pos, hotkey);

            if (ImGui::Button(param_label.c_str())) {
                p->setDirty();
            }
        } else if (auto* p = slot.template Param<core::param::ColorParam>()) {
            core::param::ColorParam::ColorType value = p->Value();
            auto color_flags = ImGuiColorEditFlags_AlphaPreview; // | ImGuiColorEditFlags_Float;
            if (ImGui::ColorEdit4(param_label.c_str(), (float*)value.data(), color_flags)) {
                p->SetValue(value);
            }

            help = "[Click] on the colored square to open a color picker.\n"
                   "[CTRL+Click] on individual component to input value.\n"
                   "[Right-Click] on the individual color widget to show options.";
        } else if (auto* p = slot.template Param<core::param::TransferFunctionParam>()) {

            ImGui::Separator();
            ImGui::Text(param_name.c_str());

            bool load_tf = false;
            param_label = "Load into Editor###editor" + param_id;
            if (p == this->tf_editor.GetActiveParameter()) {
                param_label = "Open Editor###editor" + param_id;
            }
            if (ImGui::Button(param_label.c_str())) {
                this->tf_editor.SetActiveParameter(p);
                load_tf = true;
                // Open window calling the transfer function editor callback
                const auto func = [&, this](const std::string& wn, GUIWinConfig& wc) {
                    if (wc.callback == GUIWindowManager::WindowDrawCallback::TF) {
                        wc.show = true;
                    }
                };
                this->window_manager.EnumWindows(func);
            }
            ImGui::SameLine();
            param_label = "Copy to Clipboard###clipboard" + param_id;
            if (ImGui::Button(param_label.c_str())) {
                ImGui::SetClipboardText(p->Value().c_str());
            }
            ImGui::SameLine();
            param_label = "Copy from Clipboard###fclipboard" + param_id;
            if (ImGui::Button(param_label.c_str())) {
                p->SetValue(ImGui::GetClipboardText());
                load_tf = true;
            }
            ImGui::SameLine();
            param_label = "Reset###reset" + param_id;
            if (ImGui::Button(param_label.c_str())) {
                p->SetValue("");
                load_tf = true;
            }
            ImGui::SameLine();
            ImGui::Text("JSON");

            if (p == this->tf_editor.GetActiveParameter()) {
                ImGui::TextColored(style.Colors[ImGuiCol_ButtonHovered], "Currently loaded into Editor");
            }

            ImGui::PushTextWrapPos(ImGui::GetContentRegionAvailWidth());
            ImGui::TextDisabled(p->Value().c_str());
            ImGui::PopTextWrapPos();

            // Loading new transfer function string from parameter into editor
            if (load_tf && (p == this->tf_editor.GetActiveParameter())) {
                if (!this->tf_editor.SetTransferFunction(p->Value())) {
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "[GUIRenderer] Couldn't load transfer function of parameter: %s.", param_id.c_str());
                }
            }

            ImGui::Separator();
        } else if (auto* p = slot.template Param<core::param::EnumParam>()) {
            // XXX: no UTF8 fanciness required here?
            auto map = p->getMap();
            auto key = p->Value();
            if (ImGui::BeginCombo(param_label.c_str(), map[key].PeekBuffer())) {
                auto iter = map.GetConstIterator();
                while (iter.HasNext()) {
                    auto pair = iter.Next();
                    bool isSelected = (pair.Key() == key);
                    if (ImGui::Selectable(pair.Value().PeekBuffer(), isSelected)) {
                        p->SetValue(pair.Key());
                    }
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        } else if (auto* p = slot.template Param<core::param::FlexEnumParam>()) {
            // XXX: no UTF8 fanciness required here?
            auto value = p->Value();
            if (ImGui::BeginCombo(param_label.c_str(), value.c_str())) {
                for (auto valueOption : p->getStorage()) {
                    bool isSelected = (valueOption == value);
                    if (ImGui::Selectable(valueOption.c_str(), isSelected)) {
                        p->SetValue(valueOption);
                    }
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        } else if (auto* p = slot.template Param<core::param::FloatParam>()) {
            auto value = p->Value();
            if (ImGui::InputFloat(param_label.c_str(), &value, 0.0f, 0.0f, float_format.c_str(),
                    ImGuiInputTextFlags_EnterReturnsTrue)) {
                p->SetValue(std::max(p->MinValue(), std::min(value, p->MaxValue())));
            }
        } else if (auto* p = slot.template Param<core::param::IntParam>()) {
            auto value = p->Value();
            if (ImGui::InputInt(param_label.c_str(), &value, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue)) {
                p->SetValue(std::max(p->MinValue(), std::min(value, p->MaxValue())));
            }
        } else if (auto* p = slot.template Param<core::param::Vector2fParam>()) {
            auto value = p->Value();
            if (ImGui::InputFloat2(param_label.c_str(), value.PeekComponents(), float_format.c_str(),
                    ImGuiInputTextFlags_EnterReturnsTrue)) {
                p->SetValue(value);
            }
        } else if (auto* p = slot.template Param<core::param::Vector3fParam>()) {
            auto value = p->Value();
            if (ImGui::InputFloat3(param_label.c_str(), value.PeekComponents(), float_format.c_str(),
                    ImGuiInputTextFlags_EnterReturnsTrue)) {
                p->SetValue(value);
            }
        } else if (auto* p = slot.template Param<core::param::Vector4fParam>()) {
            auto value = p->Value();
            if (ImGui::InputFloat4(param_label.c_str(), value.PeekComponents(), float_format.c_str(),
                    ImGuiInputTextFlags_EnterReturnsTrue)) {
                p->SetValue(value);
            }
        } else { // if (auto* p = slot.Param<core::param::StringParam>()) {

            // XXX: UTF8 conversion and allocation every frame is horrific inefficient.
            vislib::StringA valueString;
            vislib::UTF8Encoder::Encode(valueString, param->ValueString());

            size_t bufferLength = GUI_MAX_BUFFER_LEN; /// std::min(4096, (valueString.Length() + 1) * 2);
            char* buffer = new char[bufferLength];
            memcpy(buffer, valueString.PeekBuffer(), valueString.Length() + 1);

            if (ImGui::InputText(param_label.c_str(), buffer, bufferLength, ImGuiInputTextFlags_EnterReturnsTrue)) {
                vislib::UTF8Encoder::Decode(valueString, vislib::StringA(buffer));
                param->ParseValue(valueString);
            }
            delete[] buffer;

            help = "Press [Return] to confirm changes.";
        }

        this->HoverToolTip(param_desc, ImGui::GetID(param_label.c_str()), 1.0f);

        this->HelpMarkerToolTip(help);

        // Reset to default style
        if (param->IsGUIReadOnly()) {
            ImGui::PopItemFlag();
            ImGui::PopStyleVar();
        }
    }
}


/**
 * GUIRenderer<M, C>::drawParameterHotkey
 */
template <class M, class C>
void GUIRenderer<M, C>::drawParameterHotkey(const core::Module& mod, core::param::ParamSlot& slot) {

    auto param = slot.Parameter();
    if (!param.IsNull()) {
        if (auto* p = slot.template Param<core::param::ButtonParam>()) {
            std::string label = slot.Name().PeekBuffer();
            std::string desc = slot.Description().PeekBuffer();
            std::string keycode = p->GetKeyCode().ToString();

            ImGui::Columns(2, "hotkey_columns", false);

            ImGui::Text(label.c_str());
            this->HoverToolTip(desc); //, ImGui::GetID(keycode.c_str()), 0.5f);

            ImGui::NextColumn();

            ImGui::Text(keycode.c_str());
            this->HoverToolTip(desc); //, ImGui::GetID(keycode.c_str()), 0.5f);

            // Reset colums
            ImGui::Columns(1);

            ImGui::Separator();
        }
    }
}


/**
 * GUIRenderer<M, C>::considerModule
 */
template <class M, class C>
bool GUIRenderer<M, C>::considerModule(const std::string& modname, std::vector<std::string>& modules_list) {

    bool retval = false;
    if (modules_list.empty()) {
        retval = true;
    } else {
        for (auto mod : modules_list) {
            if (modname == mod) {
                retval = true;
                break;
            }
        }
    }
    return retval;
}


/**
 * GUIRenderer<M, C>::shutdown
 */
template <class M, class C> void GUIRenderer<M, C>::shutdown(void) {

    vislib::sys::Log::DefaultLog.WriteInfo("[GUIRenderer][shutdown] Initialising megamol core instance shutdown ...");
    this->GetCoreInstance()->Shutdown();
}


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIRENDERER_H_INCLUDED
