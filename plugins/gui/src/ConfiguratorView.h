/*
 * ConfiguratorView.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_CONFIGURATORVIEW_H_INCLUDED
#define MEGAMOL_GUI_CONFIGURATORVIEW_H_INCLUDED

#include "mmcore/CallerSlot.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/CallRenderView.h"

#include <imgui.h>

#include "GUIUtils.h"

 /// CMake exeption for the cluster "stampede2" running CentOS. (C++ filesystem support is not working?)
#ifdef GUI_USE_FILEUTILS
#    include "FileUtils.h"
#endif // GUI_USE_FILEUTILS


namespace megamol {
namespace gui {

class ConfiguratorView : public megamol::core::view::AbstractView {
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
    static const char* ClassName(void) { return "ConfiguratorView"; }

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
    ConfiguratorView();

    /**
     * Finalises an instance.
     */
    virtual ~ConfiguratorView();

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

    // VARIABLES --------------------------------------------------------------

    /** The override call */
    megamol::core::view::CallRenderView* overrideCall;

    /** The input renderview slot */
    core::CallerSlot render_view_slot;

    /** The ImGui context created and used by this ConfiguratorView */
    ImGuiContext* context;

    /** A parameter to select the style */
    core::param::ParamSlot style_param;

    /** Utils being used all over the place */
    GUIUtils utils;

    /** Additional UTF-8 glyph ranges for all ImGui fonts. */
    std::vector<ImWchar> font_utf8_ranges;

    /** Last instance time. */
    double last_instance_time; 

    // FUNCTIONS --------------------------------------------------------------

    /**
     * Validates Configurator parameters.
     */
    void validateConfigurator(void);

    /**
     * Draws the Configurator.
     *
     * @param viewport      The currently available viewport.
     * @param instanceTime  The current instance time.
     */
    bool drawConfigurator(vislib::math::Rectangle<int> viewport, double instanceTime);

    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_CONFIGURATORVIEW_H_INCLUDED