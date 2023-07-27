/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/view/MouseFlags.h"


/*
 * megamol::core::view::MOUSEFLAG_BUTTON_LEFT_DOWN
 */
const megamol::core::view::MouseFlags megamol::core::view::MOUSEFLAG_BUTTON_LEFT_DOWN = 0x00000001;


/*
 * megamol::core::view::MOUSEFLAG_BUTTON_RIGHT_DOWN
 */
const megamol::core::view::MouseFlags megamol::core::view::MOUSEFLAG_BUTTON_RIGHT_DOWN = 0x00000002;


/**
 * megamol::core::view::MOUSEFLAG_BUTTON_MIDDLE_DOWN
 */
const megamol::core::view::MouseFlags megamol::core::view::MOUSEFLAG_BUTTON_MIDDLE_DOWN = 0x00000004;


/**
 * megamol::core::view::MOUSEFLAG_BUTTON_LEFT_CHANGED
 */
const megamol::core::view::MouseFlags megamol::core::view::MOUSEFLAG_BUTTON_LEFT_CHANGED = 0x00000008;


/**
 * megamol::core::view::MOUSEFLAG_BUTTON_RIGHT_CHANGED
 */
const megamol::core::view::MouseFlags megamol::core::view::MOUSEFLAG_BUTTON_RIGHT_CHANGED = 0x00000010;


/**
 * megamol::core::view::MOUSEFLAG_BUTTON_MIDDLE_CHANGED
 */
const megamol::core::view::MouseFlags megamol::core::view::MOUSEFLAG_BUTTON_MIDDLE_CHANGED = 0x00000020;


/**
 * megamol::core::view::MOUSEFLAG_MODKEY_SHIFT_DOWN
 */
const megamol::core::view::MouseFlags megamol::core::view::MOUSEFLAG_MODKEY_SHIFT_DOWN = 0x00000040;


/**
 * megamol::core::view::MOUSEFLAG_MODKEY_CTRL_DOWN
 */
const megamol::core::view::MouseFlags megamol::core::view::MOUSEFLAG_MODKEY_CTRL_DOWN = 0x00000080;


/**
 * megamol::core::view::MOUSEFLAG_MODKEY_ALT_DOWN
 */
const megamol::core::view::MouseFlags megamol::core::view::MOUSEFLAG_MODKEY_ALT_DOWN = 0x00000100;


/**
 * megamol::core::view::MOUSEFLAG_MODKEY_SHIFT_CHANGED
 */
const megamol::core::view::MouseFlags megamol::core::view::MOUSEFLAG_MODKEY_SHIFT_CHANGED = 0x00000200;


/**
 * megamol::core::view::MOUSEFLAG_MODKEY_CTRL_CHANGED
 */
const megamol::core::view::MouseFlags megamol::core::view::MOUSEFLAG_MODKEY_CTRL_CHANGED = 0x00000400;


/**
 * megamol::core::view::MOUSEFLAG_MODKEY_ALT_CHANGED
 */
const megamol::core::view::MouseFlags megamol::core::view::MOUSEFLAG_MODKEY_ALT_CHANGED = 0x00000800;


/*
 * megamol::core::view::MouseFlagsResetAllChanged
 */
void megamol::core::view::MouseFlagsResetAllChanged(MouseFlags& flags) {
    flags = flags & (MOUSEFLAG_BUTTON_LEFT_DOWN | MOUSEFLAG_BUTTON_RIGHT_DOWN | MOUSEFLAG_BUTTON_MIDDLE_DOWN |
                        MOUSEFLAG_MODKEY_SHIFT_DOWN | MOUSEFLAG_MODKEY_CTRL_DOWN | MOUSEFLAG_MODKEY_ALT_DOWN);
}


/*
 * megamol::core::view::MouseFlagsSetFlag
 */
void megamol::core::view::MouseFlagsSetFlag(MouseFlags& flags, MouseFlags flag, bool set) {
    bool changed = false;
    switch (flag) {
    case MOUSEFLAG_BUTTON_LEFT_DOWN:
    case MOUSEFLAG_BUTTON_RIGHT_DOWN:
    case MOUSEFLAG_BUTTON_MIDDLE_DOWN:
    case MOUSEFLAG_MODKEY_SHIFT_DOWN:
    case MOUSEFLAG_MODKEY_CTRL_DOWN:
    case MOUSEFLAG_MODKEY_ALT_DOWN:
        if (((flags & flag) == flag) != set) {
            changed = true;
        }
        if (set) {
            flags |= flag;
        } else {
            flags &= ~flag;
        }
        break;
    default:
        // intentionally empty
        break;
    }
    if (changed) {
        switch (flag) {
        case MOUSEFLAG_BUTTON_LEFT_DOWN:
            flags |= MOUSEFLAG_BUTTON_LEFT_CHANGED;
            break;
        case MOUSEFLAG_BUTTON_RIGHT_DOWN:
            flags |= MOUSEFLAG_BUTTON_RIGHT_CHANGED;
            break;
        case MOUSEFLAG_BUTTON_MIDDLE_DOWN:
            flags |= MOUSEFLAG_BUTTON_MIDDLE_CHANGED;
            break;
        case MOUSEFLAG_MODKEY_SHIFT_DOWN:
            flags |= MOUSEFLAG_MODKEY_SHIFT_CHANGED;
            break;
        case MOUSEFLAG_MODKEY_CTRL_DOWN:
            flags |= MOUSEFLAG_MODKEY_CTRL_CHANGED;
            break;
        case MOUSEFLAG_MODKEY_ALT_DOWN:
            flags |= MOUSEFLAG_MODKEY_ALT_CHANGED;
            break;
        default:
            // intentionally empty
            break;
        }
    }
}
