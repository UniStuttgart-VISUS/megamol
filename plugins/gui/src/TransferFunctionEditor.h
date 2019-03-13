/*
 * TransferFunctionEditor.h
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


#include <array>
#include <vector>
#include <string>
#include <algorithm> // sort
#include <map>

#include <imgui.h>

#include "vislib/assert.h"


namespace megamol {
namespace gui {


/**
 * 1D Transfer Function Editor using ImGui.
 */
class TransferFunctionEditor {
public:

    /** Interpolstion modes. */
    enum InterpolMode { LINEAR = 0, GAUSS = 1 };

    /**
     * Ctor
     */
    TransferFunctionEditor(void);

    /**
     * Dtor
     */
    ~TransferFunctionEditor(void);

    /**
     * Draws the transfer function editor.
     */
    bool DrawTransferFunctionEditor(void);

    /**
     * Set transfer function data to use in editor.
     */
    void SetTransferFunction(std::vector<std::array<float, 5>>& data, InterpolMode imod);


    /**
     * Get current transfer function data.
     */
    std::vector<std::array<float, 5>> GetTransferFunction(void);

protected:

private:

    // VARIABLES -----------------------------------------------------------

    /** Array holding current colors and function values. */
    std::vector<std::array<float, 5>> data;

    /** Current interpolation option. */
    InterpolMode interpol_mode;

    /** Current texture size. */
    size_t tex_size;

    /** Recalculate texture data. */
    bool tex_recalc;

    /** Current texture data. */
    std::vector<ImVec4> tex_data;

    /** Currently active color channels in plot. */
    std::array<bool, 4> plot_channels;

    /** Currently slected node. */
    unsigned int point_select_node;

    /** Currently selected color channel of selected node. */
    unsigned int point_select_chan;
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_TRANSFERFUNCTIONEDITOR_H_INCLUDED
