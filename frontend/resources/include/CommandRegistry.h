/*
 * CommandRegistry.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <functional>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "KeyboardMouseInput.h"

#ifdef MEGAMOL_USE_CUESDK
#include "CUESDK.h"
#endif

namespace megamol::core::param {
class AbstractParam;
}

namespace megamol::frontend_resources {

static std::string CommandRegistry_Req_Name = "CommandRegistry";

struct Command {
    using EffectFunction = std::function<void(const Command*)>;
    std::string name;
    KeyCode key;
    std::string parent;
    enum class parent_type_c {
        PARENT_PARAM = 1,
        PARENT_GUI_HOTKEY = 2,
        PARENT_GUI_WINDOW = 3,
        PARENT_GUI_WINDOW_HOTKEY = 4
    };
    parent_type_c parent_type = parent_type_c::PARENT_PARAM;
    EffectFunction effect;

    void execute() const {
        auto f = effect;
        f(this);
    }

    Command() = default;
};

// note: effect must be recovered on the fly.
inline void to_json(nlohmann::json& j, const Command& c) {
    j = nlohmann::json{{{"name", c.name}, {"key", static_cast<int>(c.key.key)}, {"mods", c.key.mods.toInt()},
        {"parent_type", static_cast<int>(c.parent_type)}, {"parent", c.parent}}};
}

// note: effect must be recovered on the fly.
inline void from_json(const nlohmann::json& j, Command& c) {
    j.at("name").get_to(c.name);
    j.at("key").get_to(c.key.key);
    int m;
    j.at("mods").get_to(m);
    c.key.mods.fromInt(m);
    j.at("parent_type").get_to(c.parent_type);
    j.at("parent").get_to(c.parent);
}


class CommandRegistry {
public:
    CommandRegistry();

    ~CommandRegistry();

    std::string add_command(const Command& c);
    const Command get_command(const KeyCode& key) const;
    const Command get_command(const std::string& command_name);

    void remove_command_by_parent(const std::string& parent_param);
    void remove_command_by_name(const std::string& command_name);

    bool update_hotkey(const std::string& command_name, KeyCode key);
    bool remove_hotkey(KeyCode key);

    void modifiers_changed(Modifiers mod);

    bool exec_command(const std::string& command_name) const;
    bool exec_command(const KeyCode& key) const;

    const std::vector<Command> list_commands() const {
        return commands;
    }

private:
    bool is_new(const std::string& name) const {
        return command_index.find(name) == command_index.end();
    }

    void rebuild_index();

    std::string increment_name(const std::string& oldname);
    void push_command(const Command& c);
    void add_color_to_layer(const megamol::frontend_resources::Command& c);
    void remove_color_from_layer(const megamol::frontend_resources::Command& c);

    std::unordered_map<KeyCode, int> key_to_command;
    std::unordered_map<std::string, int> command_index;
    std::vector<Command> commands;

#ifdef MEGAMOL_USE_CUESDK
    static std::unordered_map<megamol::frontend_resources::Key, CorsairLedId> corsair_led_from_glfw_key;
    std::unordered_map<Modifiers, std::vector<CorsairLedColor>> key_colors;
    CorsairLedPositions* led_positions = nullptr;
    std::vector<CorsairLedColor> black_keyboard;
#endif

    Modifiers current_modifiers;
};

} // namespace megamol::frontend_resources
