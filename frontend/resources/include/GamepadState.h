/*
 * GamepadState.h
 *
 * Copyright (C) 2022 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <cstring>
#include <functional>
#include <list>
#include <string>
#include <vector>

namespace megamol {
namespace frontend_resources {

// SDL compatible XBox-like gamepad layout
// as defined in glfw3.h
struct GamepadState {
    // Axis, Button enumeration taken from the GLFW docs
    // https://www.glfw.org/docs/3.3/group__input.html
    enum class Axis : unsigned int {
        LEFT_X = 0,
        LEFT_Y = 1,
        RIGHT_X = 2,
        RIGHT_Y = 3,
        LEFT_TRIGGER = 4,
        RIGHT_TRIGGER = 5,
        LAST = RIGHT_TRIGGER,
    };
    enum class Button : unsigned int {
        A = 0,
        B = 1,
        X = 2,
        Y = 3,
        LEFT_BUMPER = 4,
        RIGHT_BUMPER = 5,
        BACK = 6,
        START = 7,
        GUIDE = 8,
        LEFT_THUMB = 9,
        RIGHT_THUMB = 10,
        DPAD_UP = 11,
        DPAD_RIGHT = 12,
        DPAD_DOWN = 13,
        DPAD_LEFT = 14,
        LAST = DPAD_LEFT,
        CROSS = A,
        CIRCLE = B,
        SQUARE = X,
        TRIANGLE = Y,
    };
    enum class ButtonIs : unsigned char {
        Released = 0,
        Pressed = 1,
    };
    enum class HatIs : unsigned char {
        CENTERED = 0,
        UP = 1,
        RIGHT = 2,
        DOWN = 4,
        LEFT = 8,
        RIGHT_UP = (RIGHT | UP),
        RIGHT_DOWN = (RIGHT | DOWN),
        LEFT_UP = (LEFT | UP),
        LEFT_DOWN = (LEFT | DOWN),
    };

    std::vector<float> axes = {};
    std::vector<unsigned char> buttons = {};
    std::vector<unsigned char> hats = {};

    std::string name;
    std::string guid;

    float axis(const Axis a) const {
        return axes[static_cast<unsigned int>(a)];
    }

    unsigned char button(const Button b) const {
        return buttons[static_cast<unsigned int>(b)];
    }

    bool pressed(const unsigned char button) const {
        return button == static_cast<unsigned char>(ButtonIs::Pressed);
    }

    bool pressed(const Button b) const {
        return pressed(button(b));
    }

    bool released(const unsigned char button) const {
        return button == static_cast<unsigned char>(ButtonIs::Released);
    }

    HatIs hat(const unsigned int index) const {
        return static_cast<HatIs>(hats[index]);
    }

    int hat_count() const {
        return hats.size();
    }

    void clear() {
        buttons.clear();
        axes.clear();
        hats.clear();
        name.clear();
        guid.clear();
    }
};

struct Connected_Gamepads {
    std::list<std::reference_wrapper<const GamepadState>> gamepads;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
