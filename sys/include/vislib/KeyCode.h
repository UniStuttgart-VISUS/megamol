/*
 * KeyCode.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_KEYCODE_H_INCLUDED
#define VISLIB_KEYCODE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/types.h"
#include "vislib/String.h"


namespace vislib {
namespace sys {


    /**
     * Utility class managing keyboard key codes.
     *
     * The key codes are managed like this:
     *
     * The bits marked by KEY_MOD are used to store information about modifier
     * keys.
     *
     * The bit KEY_SPECIAL is used to set that the code contains special
     * key code. If this flag is absent, the lowermost 8 bit are interpreted
     * as ASCII key (be aware of your current local).
     *
     * If you use an ASCII key code, be away that the 'shift' modifier might
     * not work as expected! E. g.: using 'A' as key code will require the
     * user to press the shift key to enter this value. However, the key code
     * does not contain the 'KEY_MOD_SHIFT' bit. Because of this this two
     * KeyCode object will be considdered not equal!
     */
    class KeyCode {
    public:

        /** Bit mask for modifier keys */
        static const WORD KEY_MOD       = 0xf000;

        /** The bit for the modifier key 'shift' */
        static const WORD KEY_MOD_SHIFT = 0x4000;

        /** The bit for the modifier key 'control' */
        static const WORD KEY_MOD_CTRL  = 0x2000;

        /** The bit for the modifier key 'alt' */
        static const WORD KEY_MOD_ALT   = 0x1000;

        /** The bit for special keys */
        static const WORD KEY_SPECIAL   = 0x0100;

        /** The special key 'Enter' */
        static const WORD KEY_ENTER     = 0x0101;

        /** The special key 'Escape' */
        static const WORD KEY_ESC       = 0x0102;

        /** The special key 'Tab' */
        static const WORD KEY_TAB       = 0x0103;

        /** The special arrow key 'Left' */
        static const WORD KEY_LEFT      = 0x0104;

        /** The special arrow key 'Up' */
        static const WORD KEY_UP        = 0x0105;

        /** The special arrow key 'Right' */
        static const WORD KEY_RIGHT     = 0x0106;

        /** The special arrow key 'Down' */
        static const WORD KEY_DOWN      = 0x0107;

        /** The special key 'Page up' */
        static const WORD KEY_PAGE_UP   = 0x0108;

        /** The special key 'Page down' */
        static const WORD KEY_PAGE_DOWN = 0x0109;

        /** The special key 'Home' */
        static const WORD KEY_HOME      = 0x010a;

        /** The special key 'End' */
        static const WORD KEY_END       = 0x010b;

        /** The special key 'Insert' */
        static const WORD KEY_INSERT    = 0x010c;

        /** The special key 'delete' */
        static const WORD KEY_DELETE    = 0x010d;

        /** The special key 'Backspace' */
        static const WORD KEY_BACKSPACE = 0x010e;

        /** The special key 'F1' */
        static const WORD KEY_F1        = 0x010f;

        /** The special key 'F2' */
        static const WORD KEY_F2        = 0x0110;

        /** The special key 'F3' */
        static const WORD KEY_F3        = 0x0111;

        /** The special key 'F4' */
        static const WORD KEY_F4        = 0x0112;

        /** The special key 'F5' */
        static const WORD KEY_F5        = 0x0113;

        /** The special key 'F6' */
        static const WORD KEY_F6        = 0x0114;

        /** The special key 'F7' */
        static const WORD KEY_F7        = 0x0115;

        /** The special key 'F8' */
        static const WORD KEY_F8        = 0x0116;

        /** The special key 'F9' */
        static const WORD KEY_F9        = 0x0117;

        /** The special key 'F10' */
        static const WORD KEY_F10       = 0x0118;

        /** The special key 'F11' */
        static const WORD KEY_F11       = 0x0119;

        /** The special key 'F12' */
        static const WORD KEY_F12       = 0x011a;

        /**
         * Ctor.
         *
         * @param key The key code to be stored.
         */
        KeyCode(void);

        /**
         * Ctor.
         *
         * @param key The key code to be stored.
         */
        explicit KeyCode(WORD key);

        /**
         * Ctor.
         *
         * @param src The object to clone from.
         */
        KeyCode(const KeyCode& src);

        /** Dtor. */
        ~KeyCode(void);

        /**
         * Answer whether this key uses the Alt modifier key.
         *
         * @return 'true' if this key uses the Alt modifier key.
         */
        inline bool IsAltMod(void) const {
            return (this->key & KEY_MOD_ALT) == KEY_MOD_ALT;
        }

        /**
         * Answer whether this key uses the Ctrl modifier key.
         *
         * @return 'true' if this key uses the Ctrl modifier key.
         */
        inline bool IsCtrlMod(void) const {
            return (this->key & KEY_MOD_CTRL) == KEY_MOD_CTRL;
        }

        /**
         * Answer whether this key uses the Shift modifier key.
         *
         * @return 'true' if this key uses the Shift modifier key.
         */
        inline bool IsShiftMod(void) const {
            return (this->key & KEY_MOD_SHIFT) == KEY_MOD_SHIFT;
        }

        /**
         * Answer whether this key is a special key.
         *
         * @return 'true' if this key is a special key.
         */
        inline bool IsSpecial(void) const {
            return (this->key & KEY_SPECIAL) == KEY_SPECIAL;
        }

        /**
         * Returns the key code without modifier keys.
         *
         * @return The key code without modifier keys.
         */
        inline WORD NoModKeys(void) const {
            return this->key & ~KEY_MOD;
        }

        /**
         * Generates a human-readable ASCII String representing the key code.
         *
         * @return A human-readable ASCII String
         */
        vislib::StringA ToStringA(void) const;

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return A reference to 'this'
         */
        KeyCode& operator=(WORD rhs);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return A reference to 'this'
         */
        KeyCode& operator=(const KeyCode& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand.
         *
         * @return 'true' if 'this' and 'rhs' are equal, 'false' if not.
         */
        bool operator==(const KeyCode& rhs) const;

        /**
         * Casts the key code to its numeric representation.
         *
         * @return The numeric value of the key code.
         */
        operator WORD(void) const;

    private:

        /** Normalises the stored key code */
        void normalise(void);

        /** The key code stored by this object */
        WORD key;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_KEYCODE_H_INCLUDED */

