/*
 * GUIWindows.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GUIWINDOWS_H_INCLUDED
#define MEGAMOL_GUI_GUIWINDOWS_H_INCLUDED


#include "mmcore/CoreInstance.h"
#include "mmcore/param/ParamSlot.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/TernaryParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/versioninfo.h"

#include "vislib/math/Rectangle.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "CorporateGreyStyle.h"
#include "CorporateWhiteStyle.h"
#include "FileUtils.h"
#include "TransferFunctionEditor.h"
#include "WindowManager.h"
#include "configurator/Configurator.h"

// Used for platform independent clipboard (ImGui so far only provides windows implementation)
#ifdef GUI_USE_GLFW
#    include "GLFW/glfw3.h"
#endif


namespace megamol {
namespace gui {

class GUIWindows {
public:
    /**
     * CTOR.
     */
    GUIWindows();

    /**
     * DTOR.
     */
    virtual ~GUIWindows();

    /**
     * Create ImGui context using OpenGL.
     *
     * @param core_instance     The currently available core instance.
     */
    bool CreateContext_GL(megamol::core::CoreInstance* core_instance);

    /**
     * Setup and enable ImGui context for subsequent use.
     *
     * @param viewport      The currently available viewport.
     * @param instanceTime  The current instance time.
     */
    bool PreDraw(vislib::math::Rectangle<int> viewport, double instanceTime);


    /**
     * Actual Gui windows drawing and final rednering of pushed ImGui draw commands.
     */
    bool PostDraw(void);

    /**
     * Process key events.
     */
    bool OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods);

    /**
     * Process character events.
     */
    bool OnChar(unsigned int codePoint);

    /**
     * Process mouse button events.
     */
    bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods);

    /**
     * Process mouse move events.
     */
    bool OnMouseMove(double x, double y);

    /**
     * Process mouse scroll events.
     */
    bool OnMouseScroll(double dx, double dy);

    /**
     * Return list of parameter slots provided by this class. Make available in module which uses this class.
     */
    inline const std::vector<megamol::core::param::ParamSlot*> GetParams(void) const { return this->param_slots; }

private:
    /** Available ImGui implementations */
    enum Implementation { NONE, OpenGL };

    /** Available GUI styles. */
    enum Styles {
        CorporateGray,
        CorporateWhite,
        DarkColors,
        LightColors,
    };

    /** ImGui key map assignment for text manipulation hotkeys (using last unused indices < 512) */
    enum GuiTextModHotkeys { CTRL_A = 506, CTRL_C = 507, CTRL_V = 508, CTRL_X = 509, CTRL_Y = 510, CTRL_Z = 511 };

    /** The global state (for settings to be applied before ImGui::Begin). */
    struct StateBuffer {
        std::string font_file;                 // Apply changed font file name.
        float font_size;                       // Apply changed font size.
        unsigned int font_index;               // Apply cahnged font by index.
        std::vector<ImWchar> font_utf8_ranges; // Additional UTF-8 glyph ranges for all ImGui fonts.
        bool win_save_state;                   // Flag indicating that window state should be written to parameter.
        float win_save_delay;      // Flag indicating how long to wait for saving window state since last user action.
        std::string win_delete;    // Name of the window to delete.
        double last_instance_time; // Last instance time.
        bool hotkeys_check_once;   // WORKAROUND: Check multiple hotkey assignments once.
    };

    /** The GUI hotkey array index mapping. */
    enum GuiHotkeyIndex : size_t { EXIT_PROGRAM = 0, PARAMETER_SEARCH = 1, SAVE_PROJECT = 2, INDEX_COUNT = 3 };

    // VARIABLES --------------------------------------------------------------

    /** Pointer to core isntance. */
    megamol::core::CoreInstance* core_instance;

    /** List of pointers to all paramters. */
    std::vector<megamol::core::param::ParamSlot*> param_slots;

    /** A parameter to select the style */
    megamol::core::param::ParamSlot style_param;

    /** A parameter to store the profile */
    megamol::core::param::ParamSlot state_param;

    /** A parameter for automatically start the configurator at start up */
    megamol::core::param::ParamSlot autostart_configurator;

    /** Hotkeys */
    std::array<megamol::gui::HotkeyDataType, GuiHotkeyIndex::INDEX_COUNT> hotkeys;

    /** The ImGui context created and used by this GUIWindows */
    ImGuiContext* context;

    /** The currently initialized ImGui implementation */
    Implementation impl;

    /** The window manager. */
    WindowManager window_manager;

    /** The transfer function editor. */
    TransferFunctionEditor tf_editor;

    /** The last tf param value. */
    size_t tf_hash;

    /** The tf texture id. */
    GLuint tf_texture_id;

    /** The configurator. */
    megamol::gui::configurator::Configurator configurator;

    /** Utils being used all over the place */
    megamol::gui::GUIUtils utils;

    /** File utils providing stuff interacting with files */
    megamol::gui::FileUtils file_utils;

    /** The current local state of the gui. */
    StateBuffer state;

    /** Input Widget Buffers. */
    std::map<std::string, std::string> widgtmap_text;
    std::map<std::string, int> widgtmap_int;
    std::map<std::string, float> widgtmap_float;
    std::map<std::string, vislib::math::Vector<float, 2>> widgtmap_vec2;
    std::map<std::string, vislib::math::Vector<float, 3>> widgtmap_vec3;
    std::map<std::string, vislib::math::Vector<float, 4>> widgtmap_vec4;

    /** Numer of fonts reserved for the configurator graph canvas. */
    unsigned int graph_fonts_reserved;

    // FUNCTIONS --------------------------------------------------------------

    /**
     * Creates the ImGui context indepedant of the required implementation.
     */
    bool createContext(void);

    /**
     * Creates the ImGui context indepedant of the required implementation.
     */
    bool destroyContext(void);

    /**
     * Validates GUI parameters.
     */
    void validateParameter();

    /**
     * Callback for drawing the parameter window.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawMainWindowCallback(const std::string& wn, WindowManager::WindowConfiguration& wc);

    /**
     * Draws parameters and options.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawParametersCallback(const std::string& wn, WindowManager::WindowConfiguration& wc);

    /**
     * Draws fps overlay window.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawFpsWindowCallback(const std::string& wn, WindowManager::WindowConfiguration& wc);

    /**
     * Callback for drawing font selection window.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawFontWindowCallback(const std::string& wn, WindowManager::WindowConfiguration& wc);

    /**
     * Callback for drawing the demo window.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawTFWindowCallback(const std::string& wn, WindowManager::WindowConfiguration& wc);

    /**
     * Callback for drawing the configurator.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawConfiguratorCallback(const std::string& wn, WindowManager::WindowConfiguration& wc);

    /**
     * Draws the menu bar.
     *
     * @param window_config  The configuration of the calling window.
     */
    void drawMenu(const std::string& wn, WindowManager::WindowConfiguration& wc);

    /**
     * Draws one parameter.
     *
     * @param mod   Module the paramter belongs to.
     * @param slot  The current parameter slot.
     */
    void drawParameter(const core::Module& mod, core::param::ParamSlot& slot);

    /**
     * Transfer function edit widget.
     */
    void drawTransferFunctionEdit(
        const std::string& id, const std::string& label, megamol::core::param::TransferFunctionParam& p);

    /**
     * Draws only a button parameter's hotkey.
     *
     * @param mod   Module the paramter belongs to.
     * @param slot  The current parameter slot.
     */
    void drawParameterHotkey(const core::Module& mod, core::param::ParamSlot& slot);

    /**
     * Check if module's parameters should be visible.
     */
    bool considerModule(const std::string& modname, std::vector<std::string>& modules_list);

    /**
     * Checks for multiple hotkey assignement.
     */
    void checkMultipleHotkeyAssignement(void);

    /**
     * Check if given hotkey is pressed.
     */
    bool hotkeyPressed(megamol::core::view::KeyCode keycode);

    /**
     * Shutdown megmol program.
     */
    void shutdown(void);
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIWINDOWS_H_INCLUDED
