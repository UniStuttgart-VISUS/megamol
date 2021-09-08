#include "CommandRegistry.h"

void megamol::frontend_resources::CommandRegistry::add_command(const megamol::frontend_resources::Command& c) {
    const bool command_is_new = command_index.find(c.name) == command_index.end();
    const bool key_code_unused = key_to_command.find(c.key) == key_to_command.end();
    if (command_is_new && key_code_unused) {
        push_command(c);
    } else {
        Command c2;
        if (!command_is_new) {
            c2.name = increment_name(c.name);
        }
        if (key_code_unused) {
            c2.key = c.key;
        }
        c2.param = c.param;
        push_command(c2);
    }
}

void megamol::frontend_resources::CommandRegistry::remove_command(const megamol::core::param::AbstractParam* param) {
    auto it = std::find_if(commands.begin(), commands.end(), [param] (const Command& c){return c.param == param;});
    if (it != commands.end()) {
        command_index.erase(it->name);
        if (it->key.key != Key::KEY_UNKNOWN) key_to_command.erase(it->key);
        commands.erase(it);
    }
}

void megamol::frontend_resources::CommandRegistry::update_hotkey(const std::string& command_name, KeyCode key) {
    if (!is_new(command_name)) {
        auto& c = commands[command_index[command_name]];
        const auto old_key = c.key;
        c.key = key;
        key_to_command.erase(old_key);
    }
}

std::string megamol::frontend_resources::CommandRegistry::increment_name(const std::string& oldname) {
    std::string new_name;
    std::string prefix;
    std::regex r("^(.*?)_\\d+$");
    std::cmatch m;
    if (std::regex_match(oldname.c_str(), m, r)) {
        // already have a suffix
        prefix = m[0];
    } else {
        prefix = oldname;
    }
    int cnt = 0;
    bool isnew = false;
    do {
        cnt++;
        new_name = prefix + "_" + std::to_string(cnt);
        isnew = is_new(new_name);
    } while (!isnew);
    return new_name;
}

void megamol::frontend_resources::CommandRegistry::push_command(const Command& c) {
    commands.push_back(c);
    if (c.key.key != Key::KEY_UNKNOWN) key_to_command[c.key] = static_cast<int>(commands.size() - 1);
    command_index[c.name] = static_cast<int>(commands.size() - 1);
}
