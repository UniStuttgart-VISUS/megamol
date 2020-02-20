/*
 * GUIView.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GUIVIEW_H_INCLUDED
#define MEGAMOL_GUI_GUIVIEW_H_INCLUDED

#include "mmcore/CallerSlot.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/CallRenderView.h"

#include "vislib/math/Rectangle.h"

#include "GUIUtils.h"
#include "TransferFunctionEditor.h"
#include "WindowManager.h"

/// CMake exeption for the cluster "stampede2" running CentOS. (C++ filesystem support is not working?)
#ifdef GUI_USE_FILEUTILS
#    include "FileUtils.h"
#endif // GUI_USE_FILEUTILS

#include <imgui.h>


namespace megamol {
namespace gui {

class GUIView : public megamol::core::view::AbstractView {
public:
    /** Available GUI styles. */
    enum Styles {
        CorporateGray,
        CorporateWhite,
        DarkColors,
        LightColors,
    };

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "GUIView"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description(void) { return "View that decorates a graphical user interface"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable(void) { return true; }

    /**
     * Initialises a new instance.
     */
    GUIView();

    /**
     * Finalises an instance.
     */
    virtual ~GUIView();

protected:
    // FUNCTIONS --------------------------------------------------------------

    virtual bool create() override;

    virtual void release() override;

    virtual float DefaultTime(double instTime) const override;

    virtual unsigned int GetCameraSyncNumber(void) const override;

    virtual void SerialiseCamera(vislib::Serialiser& serialiser) const override;

    virtual void DeserialiseCamera(vislib::Serialiser& serialiser) override;

    virtual bool OnRenderView(megamol::core::Call& call);

    virtual void Render(const mmcRenderViewContext& context) override;

    virtual void ResetView(void) override;

    virtual void Resize(unsigned int width, unsigned int height) override;

    virtual void UpdateFreeze(bool freeze) override;

    virtual bool OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) override;

    virtual bool OnChar(unsigned int codePoint) override;

    virtual bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    virtual bool OnMouseMove(double x, double y) override;

    virtual bool OnMouseScroll(double dx, double dy) override;

    /**
     * Unpacks the mouse coordinates, which are relative to the virtual
     * viewport size.
     *
     * @param x The x coordinate of the mouse position
     * @param y The y coordinate of the mouse position
     */
    virtual void unpackMouseCoordinates(float& x, float& y);

private:
    /** ImGui key map assignment for text manipulation hotkeys (using last unused indices < 512) */
    enum GuiTextModHotkeys { CTRL_A = 506, CTRL_C = 507, CTRL_V = 508, CTRL_X = 509, CTRL_Y = 510, CTRL_Z = 511 };

    /** The global state (for settings to be applied before ImGui::Begin). */
    struct StateBuffer {
        std::string font_file;                 // Apply changed font file name.
        float font_size;                       // Apply changed font size.
        int font_index;                        // Apply cahnged font by index.
        std::vector<ImWchar> font_utf8_ranges; // Additional UTF-8 glyph ranges for all ImGui fonts.
        bool win_save_state;                   // Flag indicating that window state should be written to parameter.
        float win_save_delay;      // Flag indicating how long to wait for saving window state since last user action.
        std::string win_delete;    // Name of the window to delete.
        double last_instance_time; // Last instance time.
        bool hotkeys_check_once;   // WORKAROUND: Check multiple hotkey assignments once.
    };


    // VARIABLES --------------------------------------------------------------

    /** The override call */
    megamol::core::view::CallRenderView* overrideCall;

    /** The input renderview slot */
    core::CallerSlot render_view_slot;

    /** A parameter to select the style */
    core::param::ParamSlot style_param;

    /** A parameter to store the profile */
    core::param::ParamSlot state_param;

    /** The ImGui context created and used by this GUIView */
    ImGuiContext* context;

    /** The window manager. */
    WindowManager window_manager;

    /** The transfer function editor. */
    TransferFunctionEditor tf_editor;

    /** Utils being used all over the place */
    GUIUtils utils;

    /** The current local state of the gui. */
    StateBuffer state;

    /** Set focus to parmeter search text input. */
    bool setParameterSearchFocus;
    /** Current parameter search string. */
    std::string parameterSearchString;
    /** Show parameter search window. */
    // bool showParameterSearchWindow;

    /** Input Widget Buffers. */
    std::map<std::string, std::string> widgtmap_text;
    std::map<std::string, int> widgtmap_int;
    std::map<std::string, float> widgtmap_float;
    std::map<std::string, vislib::math::Vector<float, 2>> widgtmap_vec2;
    std::map<std::string, vislib::math::Vector<float, 3>> widgtmap_vec3;
    std::map<std::string, vislib::math::Vector<float, 4>> widgtmap_vec4;

    // FUNCTIONS --------------------------------------------------------------

    /**
     * Validates GUI parameters.
     */
    void validateGUI();

    /**
     * Draws the GUI.
     *
     * @param viewport      The currently available viewport.
     * @param instanceTime  The current instance time.
     */
    bool drawGUI(vislib::math::Rectangle<int> viewport, double instanceTime);

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
     * Returns true if search string is found in source as a case insensitive substring.
     *
     * @param source   The string to search in.
     * @param search   The string to search for in the source.
     */
    bool findCaseInsensitiveSubstring(const std::string& source, const std::string& search);

    /**
     * Check if module's parameters should be visible.
     */
    bool considerModule(const std::string& modname, std::vector<std::string>& modules_list);

    /**
     * Checks for multiple hotkey assignement.
     */
    void checkMultipleHotkeyAssignement(void);

    /**
     * Shutdown megmol program.
     */
    void shutdown(void);

    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIVIEW_H_INCLUDED