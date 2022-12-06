/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "vislib/types.h"

namespace megamol::core::view {

/**
 * Type for storing mouse flags
 */
typedef UINT32 MouseFlags;

/**
 * Indicates that the left mouse button is pressed
 */
extern const MouseFlags MOUSEFLAG_BUTTON_LEFT_DOWN;

/**
 * Indicates that the right mouse button is pressed
 */
extern const MouseFlags MOUSEFLAG_BUTTON_RIGHT_DOWN;

/**
 * Indicates that the middle mouse button is pressed
 */
extern const MouseFlags MOUSEFLAG_BUTTON_MIDDLE_DOWN;

/**
 * Indicates that the left mouse button changed its state
 */
extern const MouseFlags MOUSEFLAG_BUTTON_LEFT_CHANGED;

/**
 * Indicates that the right mouse button changed its state
 */
extern const MouseFlags MOUSEFLAG_BUTTON_RIGHT_CHANGED;

/**
 * Indicates that the middle mouse button changed its state
 */
extern const MouseFlags MOUSEFLAG_BUTTON_MIDDLE_CHANGED;

/**
 * Indicates that the shift modifier key is pressed
 */
extern const MouseFlags MOUSEFLAG_MODKEY_SHIFT_DOWN;

/**
 * Indicates that the ctrl modifier key is pressed
 */
extern const MouseFlags MOUSEFLAG_MODKEY_CTRL_DOWN;

/**
 * Indicates that the alt modifier key is pressed
 */
extern const MouseFlags MOUSEFLAG_MODKEY_ALT_DOWN;

/**
 * Indicates that the shift modifier key changed its state
 */
extern const MouseFlags MOUSEFLAG_MODKEY_SHIFT_CHANGED;

/**
 * Indicates that the ctrl modifier key changed its state
 */
extern const MouseFlags MOUSEFLAG_MODKEY_CTRL_CHANGED;

/**
 * Indicates that the alt modifier key changed its state
 */
extern const MouseFlags MOUSEFLAG_MODKEY_ALT_CHANGED;

/**
 * Resets all 'changed' flags
 *
 * @param flags The mouse flags to be changed
 */
void MouseFlagsResetAllChanged(MouseFlags& flags);

/**
 * Sets the flag 'flag' in 'flags'. Also sets the corresponding changed
 * flag if required.
 *
 * @param flags The mouse flags to be changed
 * @param flag The flag to (re-)set
 * @param set The new state for the flag.
 */
void MouseFlagsSetFlag(MouseFlags& flags, MouseFlags flag, bool set = true);

/**
 * Clears the flag 'flag' in 'flags'. Also sets the corresponding changed
 * flag if required.
 *
 * @param flags The mouse flags to be changed
 * @param flag The flag to clear
 */
inline void MouseFlagsClearFlag(MouseFlags& flags, MouseFlags flag) {
    MouseFlagsSetFlag(flags, flag, false);
}

} // namespace megamol::core::view
