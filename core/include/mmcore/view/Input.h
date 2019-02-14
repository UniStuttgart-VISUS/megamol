/*
 * Input.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_INPUT_H_INCLUDED
#define MEGAMOLCORE_INPUT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <bitset>
#include <type_traits>

namespace megamol {
namespace core {
namespace view {

enum class Key : int {
    KEY_UNKNOWN = -1,
    KEY_SPACE = 32,
    KEY_APOSTROPHE = 39, /* ' */
    KEY_COMMA = 44,      /* , */
    KEY_MINUS = 45,      /* - */
    KEY_PERIOD = 46,     /* . */
    KEY_SLASH = 47,      /* / */
    KEY_0 = 48,
    KEY_1 = 49,
    KEY_2 = 50,
    KEY_3 = 51,
    KEY_4 = 52,
    KEY_5 = 53,
    KEY_6 = 54,
    KEY_7 = 55,
    KEY_8 = 56,
    KEY_9 = 57,
    KEY_SEMICOLON = 59, /* ; */
    KEY_EQUAL = 61,     /* = */
    KEY_A = 65,
    KEY_B = 66,
    KEY_C = 67,
    KEY_D = 68,
    KEY_E = 69,
    KEY_F = 70,
    KEY_G = 71,
    KEY_H = 72,
    KEY_I = 73,
    KEY_J = 74,
    KEY_K = 75,
    KEY_L = 76,
    KEY_M = 77,
    KEY_N = 78,
    KEY_O = 79,
    KEY_P = 80,
    KEY_Q = 81,
    KEY_R = 82,
    KEY_S = 83,
    KEY_T = 84,
    KEY_U = 85,
    KEY_V = 86,
    KEY_W = 87,
    KEY_X = 88,
    KEY_Y = 89,
    KEY_Z = 90,
    KEY_LEFT_BRACKET = 91,  /* [ */
    KEY_BACKSLASH = 92,     /* \ */
    KEY_RIGHT_BRACKET = 93, /* ] */
    KEY_GRAVE_ACCENT = 96,  /* ` */
    //KEY_a = 97,
    //KEY_b = 98,
    //KEY_c = 99,
    //KEY_d = 100,
    //KEY_e = 101,
    //KEY_f = 102,
    //KEY_g = 103,
    //KEY_h = 104,
    //KEY_i = 105,
    //KEY_j = 106,
    //KEY_k = 107,
    //KEY_l = 108,
    //KEY_m = 109,
    //KEY_n = 110,
    //KEY_o = 111,
    //KEY_p = 112,
    //KEY_q = 113,
    //KEY_r = 114,
    //KEY_s = 115,
    //KEY_t = 116,
    //KEY_u = 117,
    //KEY_v = 118,
    //KEY_w = 119,
    //KEY_x = 120,
    //KEY_y = 121,
    //KEY_z = 122,
    KEY_WORLD_1 = 161,      /* non-US #1 */
    KEY_WORLD_2 = 162,      /* non-US #2 */
    KEY_ESCAPE = 256,
    KEY_ENTER = 257,
    KEY_TAB = 258,
    KEY_BACKSPACE = 259,
    KEY_INSERT = 260,
    KEY_DELETE = 261,
    KEY_RIGHT = 262,
    KEY_LEFT = 263,
    KEY_DOWN = 264,
    KEY_UP = 265,
    KEY_PAGE_UP = 266,
    KEY_PAGE_DOWN = 267,
    KEY_HOME = 268,
    KEY_END = 269,
    KEY_CAPS_LOCK = 280,
    KEY_SCROLL_LOCK = 281,
    KEY_NUM_LOCK = 282,
    KEY_PRINT_SCREEN = 283,
    KEY_PAUSE = 284,
    KEY_F1 = 290,
    KEY_F2 = 291,
    KEY_F3 = 292,
    KEY_F4 = 293,
    KEY_F5 = 294,
    KEY_F6 = 295,
    KEY_F7 = 296,
    KEY_F8 = 297,
    KEY_F9 = 298,
    KEY_F10 = 299,
    KEY_F11 = 300,
    KEY_F12 = 301,
    KEY_F13 = 302,
    KEY_F14 = 303,
    KEY_F15 = 304,
    KEY_F16 = 305,
    KEY_F17 = 306,
    KEY_F18 = 307,
    KEY_F19 = 308,
    KEY_F20 = 309,
    KEY_F21 = 310,
    KEY_F22 = 311,
    KEY_F23 = 312,
    KEY_F24 = 313,
    KEY_F25 = 314,
    KEY_KP_0 = 320,
    KEY_KP_1 = 321,
    KEY_KP_2 = 322,
    KEY_KP_3 = 323,
    KEY_KP_4 = 324,
    KEY_KP_5 = 325,
    KEY_KP_6 = 326,
    KEY_KP_7 = 327,
    KEY_KP_8 = 328,
    KEY_KP_9 = 329,
    KEY_KP_DECIMAL = 330,
    KEY_KP_DIVIDE = 331,
    KEY_KP_MULTIPLY = 332,
    KEY_KP_SUBTRACT = 333,
    KEY_KP_ADD = 334,
    KEY_KP_ENTER = 335,
    KEY_KP_EQUAL = 336,
    KEY_LEFT_SHIFT = 340,
    KEY_LEFT_CONTROL = 341,
    KEY_LEFT_ALT = 342,
    KEY_LEFT_SUPER = 343,
    KEY_RIGHT_SHIFT = 344,
    KEY_RIGHT_CONTROL = 345,
    KEY_RIGHT_ALT = 346,
    KEY_RIGHT_SUPER = 347,
    KEY_MENU = 348
};

enum class KeyAction : int { PRESS = 0, REPEAT, RELEASE };

enum class MouseButton : int {
    BUTTON_1 = 0,
    BUTTON_2 = 1,
    BUTTON_3 = 2,
    BUTTON_4 = 3,
    BUTTON_5 = 4,
    BUTTON_6 = 5,
    BUTTON_7 = 6,
    BUTTON_8 = 7,
    BUTTON_LEFT = BUTTON_1,
    BUTTON_MIDDLE = BUTTON_3,
    BUTTON_RIGHT = BUTTON_2,
};

enum class MouseButtonAction : int { PRESS = 0, RELEASE };

enum class Modifier : int { SUPER = 8, ALT = 4, CTRL = 2, SHIFT = 1 };

class Modifiers {
    typedef std::bitset<3> Bits;
    Bits bits;

public:
    Modifiers() {}

    Modifiers(Modifier mod) : bits(static_cast<typename std::underlying_type<Modifier>::type>(mod)) {}

    explicit Modifiers(int bits) : bits(bits) {}

    inline int toInt() { return static_cast<int>(bits.to_ulong()); }

    inline bool none() const { return bits.none(); }

    inline bool test(Modifier mod) const { return (bits & Modifiers(mod).bits).any(); }

    inline Modifiers& reset(Modifier mod) {
        Modifiers mask(mod);
        bits ^= mask.bits;
        return *this;
    }

    inline Modifiers& set(Modifier mod) {
        Modifiers mask(mod);
        bits |= mask.bits;
        return *this;
    }

    inline Modifiers& operator|=(const Modifiers& other) {
        bits |= other.bits;
        return *this;
    }

    inline Modifiers& operator&=(const Modifiers& other) {
        bits &= other.bits;
        return *this;
    }

    inline Modifiers& operator^=(const Modifiers& other) {
        bits = other.bits;
        return *this;
    }
};

inline Modifiers operator|(Modifiers lhs, Modifiers rhs) {
    Modifiers ans = lhs;
    return (ans |= rhs);
}

inline Modifiers operator&(Modifiers lhs, Modifiers rhs) {
    Modifiers ans = lhs;
    return (ans &= rhs);
}

inline Modifiers operator^(Modifiers lhs, Modifiers rhs) {
    Modifiers ans = lhs;
    return (ans ^= rhs);
}

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_INPUT_H_INCLUDED */
