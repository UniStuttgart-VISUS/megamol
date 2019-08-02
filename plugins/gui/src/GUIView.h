/*
 * GUIView.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/view/AbstractView.h"

#include "vislib/math/Rectangle.h"

#include "Popup.h"
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

private:
    // ENUMS ------------------------------------------------------------------

    /** ImGui key map assignment for text manipulation hotkeys (using last unused indices < 512) */
    enum GuiTextModHotkeys { CTRL_A = 506, CTRL_C = 507, CTRL_V = 508, CTRL_X = 509, CTRL_Y = 510, CTRL_Z = 511 };

    // VARIABLES --------------------------------------------------------------

    /** The input renderview slot */
    core::CallerSlot renderViewSlot;

    /** A parameter to select the style */
    core::param::ParamSlot styleParam;

    /** A parameter to store the profile */
    core::param::ParamSlot stateParam;

    /** The ImGui context created and used by this GUIView */
    ImGuiContext* context;

    /** The window manager. */
    WindowManager windowManager;

    /** The transfer function editor. */
    TransferFunctionEditor tfEditor;

    /** A popup being used all over the place */
    Popup popup;

    /** Last instance time.  */
    double lastInstanceTime;

    /** Additional UTF-8 glyph ranges for ImGui fonts. */
    std::vector<ImWchar> fontUtf8Ranges;

    /** Saving the last given project filename. */
    std::string projectFilename;

    // Window state buffer variables: -----------------------------------------

    /** File name of font file to load. */
    std::string newFontFilenameToLoad;

    /** Font size of font to load. */
    float newFontSizeToLoad;

    /** Load font by index. */
    int newFontIndexToLoad;

    /** Name of window to delete. */
    std::string windowToDelete;

    /** Flag indicating that window state should be written to parameter. */
    bool saveState;
    float saveStateDelay;

    /** WORKAROUND: Check multiple hotkey assignment once. */
    bool checkHotkeysOnce;

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
    void drawMainWindowCallback(const std::string& window_name, WindowManager::WindowConfiguration& window_config);

    /**
     * Draws parameters and options.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawParametersCallback(const std::string& window_name, WindowManager::WindowConfiguration& window_config);

    /**
     * Draws fps overlay window.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawFpsWindowCallback(const std::string& window_name, WindowManager::WindowConfiguration& window_config);

    /**
     * Callback for drawing font selection window.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawFontWindowCallback(const std::string& window_name, WindowManager::WindowConfiguration& window_config);

    /**
     * Callback for drawing the demo window.
     *
     * @param window_name    The label of the calling window.
     * @param window_config  The configuration of the calling window.
     */
    void drawTFWindowCallback(const std::string& window_name, WindowManager::WindowConfiguration& window_config);

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
