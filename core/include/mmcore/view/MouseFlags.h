/*
 * MouseFlags.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MOUSEFLAGS_H_INCLUDED
#define MEGAMOLCORE_MOUSEFLAGS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/types.h"


namespace megamol {
namespace core {
namespace view {

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

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MOUSEFLAGS_H_INCLUDED */
