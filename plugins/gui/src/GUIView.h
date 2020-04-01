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

#include "vislib/sys/Log.h"

#include "GUIWindows.h"


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

    virtual bool OnRenderView(megamol::core::Call& call);

    virtual void Render(const mmcRenderViewContext& context) override;

    virtual void ResetView(void) override;

    virtual void Resize(unsigned int width, unsigned int height) override;

    virtual void UpdateFreeze(bool freeze) override;

    virtual bool OnKey(megamol::core::view::Key key, megamol::core::view::KeyAction action,
        megamol::core::view::Modifiers mods) override;

    virtual bool OnChar(unsigned int codePoint) override;

    virtual bool OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action,
        megamol::core::view::Modifiers mods) override;

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
    // VARIABLES --------------------------------------------------------------

    /** The override call */
    megamol::core::view::CallRenderView* overrideCall;

    /** The input renderview slot */
    megamol::core::CallerSlot render_view_slot;

    /** The gui */
    megamol::gui::GUIWindows gui;

    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIVIEW_H_INCLUDED
