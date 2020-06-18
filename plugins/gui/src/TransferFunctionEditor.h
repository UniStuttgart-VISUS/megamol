/*
 * TransferFunctionEditor.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_TRANSFERFUNCTIONEDITOR_INCLUDED
#define MEGAMOL_GUI_TRANSFERFUNCTIONEDITOR_INCLUDED


#include "ColorPalettes.h"

#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/view/TransferFunction.h"

#include "GUIUtils.h"

#include <cmath>
#include <iomanip>
#include <sstream>


namespace megamol {
namespace gui {

namespace configurator {

// Forward declarations
class Parameter;
typedef std::shared_ptr<Parameter> ParamPtrType;

} // namespace configurator


/**
 * 1D Transfer Function Editor.
 */
class TransferFunctionEditor {
public:
    TransferFunctionEditor(void);

    ~TransferFunctionEditor(void) = default;

    /**
     * Draws the transfer function editor.
     */
    bool Draw(bool connected_parameter_mode);

    /**
     * Set transfer function data to use in editor.
     *
     * @param tfs The transfer function encoded as string in JSON format.
     *
     * @return True if string was successfully converted into transfer function data, false otherwise.
     */
    void SetTransferFunction(const std::string& tfs, bool connected_parameter_mode);

    /**
     * Get current transfer function data.
     *
     * @return The transfer function encoded as string in JSON format
     */
    bool GetTransferFunction(std::string& tfs);

    /**
     * Set the currently connected parameter.
     */
    void SetConnectedParameter(configurator::ParamPtrType param_ptr);

    /**
     * Get currently connected parameter.
     */
    inline const configurator::ParamPtrType GetConnectedParameter(void) const { return this->connected_parameter_ptr; }

    /**
     * Returns true if editor is in minimized view.
     */
    inline bool IsMinimized(void) const { return !this->showOptions; }

    /**
     * Set minimized view.
     */
    inline void SetMinimized(bool minimized) { this->showOptions = !minimized; }

    /**
     * Returns true if editor is in vertical view.
     */
    inline bool IsVertical(void) const { return this->flip_xy; }

    /**
     * Set vertical view.
     */
    inline void SetVertical(bool vertical) { this->flip_xy = vertical; }

    /**
     * Return texture id of horizontal texture.
     */
    GLuint GetHorizontalTexture(void) { return this->texture_id_horiz; }

private:
    /** The global input widget state buffer. */
    struct WidgetBuffer {
        float min_range;
        float max_range;
        float gauss_sigma;
        float range_value;
        int tex_size;
    };

    // VARIABLES -----------------------------------------------------------

    /** Utils being used all over the place */
    GUIUtils utils;

    /** The currently active parameter whose transfer function is currently loaded into this editor. */
    configurator::ParamPtrType connected_parameter_ptr;

    /** Array holding current colors and function values. */
    megamol::core::param::TransferFunctionParam::TFNodeType nodes;

    /** Min/Max intervall the data should be mapped. */
    std::array<float, 2> range;
    std::array<float, 2> last_range;

    /** Flag indicating if propagated range should be overwriten by editor */
    bool range_overwrite;

    /** Current interpolation option. */
    megamol::core::param::TransferFunctionParam::InterpolationMode mode;

    /** Current texture size. */
    UINT textureSize;

    /** Indicating modified transfer function. Recalculate texture data. */
    bool textureInvalid;

    /** Indicates whether changes are already applied or not. */
    bool pendingChanges;

    /** Current texture data. */
    std::vector<float> texturePixels;

    /** OpenGL Texture IDs. */
    GLuint texture_id_vert;
    GLuint texture_id_horiz;

    /** Currently active color channels in plot. */
    std::array<bool, 4> activeChannels;

    /** Currently selected node. */
    unsigned int currentNode;

    /** Currently selected color channel of selected node. */
    unsigned int currentChannel;

    /** Offset from center of point to initial drag position. */
    ImVec2 currentDragChange;

    /** Flag for applying all changes immediately. */
    bool immediateMode;

    /** Flag indicating if all options should be shown*/
    bool showOptions;

    /** The global input widget state buffer. */
    WidgetBuffer widget_buffer;

    /** Legend alignment flag. */
    bool flip_xy;

    // FUNCTIONS -----------------------------------------------------------

    void drawTextureBox(const ImVec2& size, bool flip_xy);

    void drawScale(const ImVec2& pos, const ImVec2& size, bool flip_xy);

    void drawFunctionPlot(const ImVec2& size);

    void createTexture(GLuint& inout_id, GLsizei width, GLsizei height, float* data) const;
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_TRANSFERFUNCTIONEDITOR_INCLUDED
