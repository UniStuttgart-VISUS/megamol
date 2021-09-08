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
    void add_command(const Command& c);

    void remove_command(const megamol::core::param::AbstractParam* param);

    void update_hotkey(const std::string& command_name, KeyCode key);


private:

    bool is_new(const std::string& name) {
        return command_index.find(name) == command_index.end();
    }

    std::string increment_name(const std::string& oldname);
    void push_command(const Command& c);

    std::unordered_map<KeyCode, int> key_to_command;
    std::unordered_map<std::string, int> command_index;
    std::vector<Command> commands;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
