/*
 * LinearTransferFunctionEditor.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#ifndef MEGAMOL_GUI_TRANSFERFUNCTIONEDITOR_H_INCLUDED
#define MEGAMOL_GUI_TRANSFERFUNCTIONEDITOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "mmcore/param/LinearTransferFunctionParam.h"
#include "mmcore/view/LinearTransferFunction.h"

#include "vislib/sys/Log.h"

#include <algorithm> // sort
#include <array>
#include <cassert>
#include <cmath> // sqrtf
#include <imgui.h>
#include <map>
#include <sstream> // stringstream
#include <string>
#include <vector>

#include <imgui.h>
#include "GUIUtility.h"


namespace megamol {
namespace gui {


/**
 * 1D Transfer Function Editor using ImGui.
 */
class LinearTransferFunctionEditor : public GUIUtility {
public:
    /**
     * Draws the transfer function editor.
     */
    bool DrawTransferFunctionEditor(void);

    /**
     * Set transfer function data to use in editor.
     *
     * @param tfs The transfer function encoded as string in JSON format.
     *
     * @return True if string was successfully converted into transfer function data, false otherwise.
     */
    bool SetTransferFunction(const std::string& in_tfs);

    /**
     * Get current transfer function data.
     *
     * @return The transfer function encoded as string in JSON format
     */
    bool GetTransferFunction(std::string& out_tfs);

protected:
    /**
     * Ctor
     */
    LinearTransferFunctionEditor(void);

    /**
     * Dtor
     */
    ~LinearTransferFunctionEditor(void);

private:
    // VARIABLES -----------------------------------------------------------

    /** Array holding current colors and function values. */
    megamol::core::param::LinearTransferFunctionParam::TFType data;

    /** Current interpolation option. */
    megamol::core::param::LinearTransferFunctionParam::InterpolationMode interpol_mode;

    /** Current texture size. */
    UINT tex_size;

    /** Indicating modified transfer function. Recalculate texture data. */
    bool tex_modified;

    /** Current texture data. */
    std::vector<float> tex_data;

    /** Currently active color channels in plot. */
    std::array<bool, 4> plot_channels;

    /** Currently slected node. */
    unsigned int point_select_node;

    /** Currently selected color channel of selected node. */
    unsigned int point_select_chan;

    /** Offset from center of point to initial drag position. */
    ImVec2 point_select_delta;

    /** Flag for applying all changes immediately. */
    bool imm_apply;
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_TRANSFERFUNCTIONEDITOR_H_INCLUDED
