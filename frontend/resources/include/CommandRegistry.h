/*
 * CommandRegistry.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <regex>

#include "KeyboardMouseInput.h"

#ifdef CUESDK_ENABLED
#define CORSAIR_LIGHTING_SDK_DISABLE_DEPRECATION_WARNINGS
#include "CUESDK.h"
#endif

namespace megamol::core::param {
    class AbstractParam;
}

namespace megamol {
namespace frontend_resources {

static std::string CommandRegistry_Req_Name = "CommandRegistry";

struct Command {
    std::string name;
    KeyCode key;
    megamol::core::param::AbstractParam* param;
};

class CommandRegistry {
public:
    CommandRegistry();

    ~CommandRegistry();

    void add_command(const Command& c);

    void remove_command(const megamol::core::param::AbstractParam* param);

    void update_hotkey(const std::string& command_name, KeyCode key);

    void modifiers_changed(Modifiers mod);

    megamol::core::param::AbstractParam* param_from_keycode(KeyCode key);

private:

    bool is_new(const std::string& name) {
        return command_index.find(name) == command_index.end();
    }

    std::string increment_name(const std::string& oldname);
    void push_command(const Command& c);
    void add_color_to_layer(const megamol::frontend_resources::Command& c);
    void remove_color_from_layer(const megamol::frontend_resources::Command& c);

    std::unordered_map<KeyCode, int> key_to_command;
    std::unordered_map<std::string, int> command_index;
    std::vector<Command> commands;

#ifdef CUESDK_ENABLED
    static std::unordered_map<megamol::frontend_resources::Key, CorsairLedId> corsair_led_from_glfw_key;
    std::unordered_map<Modifiers, std::vector<CorsairLedColor>> key_colors;
    CorsairLedPositions* led_positions = nullptr;
    std::vector<CorsairLedColor> black_keyboard;
#endif

    Modifiers current_modifiers;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
