/*
 * KeyboardMouse_Events.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <tuple>
#include <vector>

#include "KeyboardMouseInput.h"

namespace megamol {
namespace input_events {

struct KeyboardState {
    std::vector<std::tuple<Key, KeyAction, Modifiers>> keys;
    std::vector<unsigned int> codepoints;
};
struct KeyboardEvents {
    std::vector<std::tuple<Key, KeyAction, Modifiers>> key_events;
    std::vector<unsigned int> codepoint_events;

    KeyboardState previous_state;

    void apply_state() {
        this->previous_state.keys = key_events;
        this->previous_state.codepoints = codepoint_events;
    }
    void clear() {
        apply_state();

        key_events.clear();
        codepoint_events.clear();
    }
};

struct MouseState {
    std::vector<std::tuple<MouseButton, MouseButtonAction, Modifiers>> buttons;
    double x_cursor_position = 0.0;
    double y_cursor_position = 0.0;
    bool entered = false;
    double x_scroll = 0.0;
    double y_scroll = 0.0;
};
struct MouseEvents {
    std::vector<std::tuple<MouseButton, MouseButtonAction, Modifiers>> buttons_events;
    std::vector<std::tuple<double, double>> position_events;
    std::vector<bool> enter_events;
    std::vector<std::tuple<double, double>> scroll_events;

    MouseState previous_state;

	void apply_state() {
        this->previous_state.buttons = buttons_events;

		if (position_events.size()) {
			this->previous_state.x_cursor_position = std::get<0>(position_events.back());
			this->previous_state.y_cursor_position = std::get<1>(position_events.back());
        }

		if (enter_events.size())
			this->previous_state.entered = enter_events.back();

		if (scroll_events.size()) {
			this->previous_state.x_scroll = std::get<0>(scroll_events.back());
			this->previous_state.y_scroll = std::get<1>(scroll_events.back());
		}
	}
    void clear() {
        apply_state();

        buttons_events.clear();
        position_events.clear();
        enter_events.clear();
        scroll_events.clear();
    }
};


namespace input = input_events;

} /* end namespace input_events */
} /* end namespace megamol */
