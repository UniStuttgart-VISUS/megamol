#ifndef MEGAMOL_GUI_GUIRENDERER_H_INCLUDED
#define MEGAMOL_GUI_GUIRENDERER_H_INCLUDED

#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer2DModule.h"

#include <unordered_map>

namespace megamol {
namespace gui {

class GUIRenderer : public core::view::Renderer2DModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName(void) { return "GUIRenderer"; }

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
    virtual bool create() override;

    virtual void release() override;

    virtual bool Render(core::view::CallRender2D& call) override;

    virtual bool GetExtents(core::view::CallRender2D& call) { return false; }

    virtual bool OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) override;

    virtual bool OnChar(unsigned int codePoint) override;

    virtual bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    virtual bool OnMouseMove(double x, double y) override;

    virtual bool OnMouseScroll(double dx, double dy) override;

private:
    /**
     * Draws the main menu bar.
     */
    void drawMainMenu();

    /**
     * Draws a parameter window.
     */
    void drawParameterWindow();

    /**
     * Draws a parameter for the parameter window.
     */
    void drawParameter(const core::Module& mod, const core::param::ParamSlot& slot);

    double lastViewportTime;

    bool parameterWindowActive;
    std::unordered_map<const core::param::ParamSlot*, char[256]> parameterStrings;
};

} // end namespace gui
} // end namespace megamol

#endif // MEGAMOL_GUI_GUIRENDERER_H_INCLUDED