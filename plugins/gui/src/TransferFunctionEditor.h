/*
 * TransferFunctionEditor.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/view/TransferFunction.h"

#include "vislib/sys/Log.h"

#include <cmath>
#include <string>
#include <vector>

#include <imgui.h>
#include "Popup.h"


namespace megamol {
namespace gui {

/**
 * 1D Transfer Function Editor.
 */
class TransferFunctionEditor : public Popup {
public:
    TransferFunctionEditor(void);

    ~TransferFunctionEditor(void) = default;

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
    bool SetTransferFunction(const std::string& tfs);

    /**
     * Get current transfer function data.
     *
     * @return The transfer function encoded as string in JSON format
     */
    bool GetTransferFunction(std::string& tfs);

    /**
     * Set the currently active parameter.
     */
    void SetActiveParameter(core::param::TransferFunctionParam* param) { this->activeParameter = param; }

    /**
     * Get the currently active parameter.
     */
    core::param::TransferFunctionParam* GetActiveParameter(void) { return this->activeParameter; }

private:
    // VARIABLES -----------------------------------------------------------

    /** The currently active parameter whose transfer function is currently loaded into this editor. */
    core::param::TransferFunctionParam* activeParameter;

    /** Array holding current colors and function values. */
    megamol::core::param::TransferFunctionParam::TFDataType data;

    /** Min/Max intervall the data should be mapped. */
    std::array<float, 2> range;

    /** Current interpolation option. */
    megamol::core::param::TransferFunctionParam::InterpolationMode mode;

    /** Current texture size. */
    UINT textureSize;

    /** Indicating modified transfer function. Recalculate texture data. */
    bool textureInvalid;

    /** Current texture data. */
    std::vector<float> texturePixels;

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
};

} // namespace gui
} // namespace megamol
