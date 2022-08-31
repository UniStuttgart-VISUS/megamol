#include "CommandRegistry.h"

#include <iostream>

#ifdef MEGAMOL_USE_CUESDK
std::unordered_map<megamol::frontend_resources::Key, CorsairLedId>
    megamol::frontend_resources::CommandRegistry::corsair_led_from_glfw_key{
        {megamol::frontend_resources::Key::KEY_ESCAPE, CLK_Escape}, {megamol::frontend_resources::Key::KEY_F1, CLK_F1},
        {megamol::frontend_resources::Key::KEY_F2, CLK_F2}, {megamol::frontend_resources::Key::KEY_F3, CLK_F3},
        {megamol::frontend_resources::Key::KEY_F4, CLK_F4}, {megamol::frontend_resources::Key::KEY_F5, CLK_F5},
        {megamol::frontend_resources::Key::KEY_F6, CLK_F6}, {megamol::frontend_resources::Key::KEY_F7, CLK_F7},
        {megamol::frontend_resources::Key::KEY_F8, CLK_F8}, {megamol::frontend_resources::Key::KEY_F9, CLK_F9},
        {megamol::frontend_resources::Key::KEY_F10, CLK_F10}, {megamol::frontend_resources::Key::KEY_F11, CLK_F11},
        {megamol::frontend_resources::Key::KEY_F12, CLK_F12},
        {megamol::frontend_resources::Key::KEY_GRAVE_ACCENT, CLK_GraveAccentAndTilde},
        {megamol::frontend_resources::Key::KEY_1, CLK_1}, {megamol::frontend_resources::Key::KEY_2, CLK_2},
        {megamol::frontend_resources::Key::KEY_3, CLK_3}, {megamol::frontend_resources::Key::KEY_4, CLK_4},
        {megamol::frontend_resources::Key::KEY_5, CLK_5}, {megamol::frontend_resources::Key::KEY_6, CLK_6},
        {megamol::frontend_resources::Key::KEY_7, CLK_7}, {megamol::frontend_resources::Key::KEY_8, CLK_8},
        {megamol::frontend_resources::Key::KEY_9, CLK_9}, {megamol::frontend_resources::Key::KEY_0, CLK_0},
        {megamol::frontend_resources::Key::KEY_MINUS, CLK_MinusAndUnderscore},
        {megamol::frontend_resources::Key::KEY_TAB, CLK_Tab}, {megamol::frontend_resources::Key::KEY_Q, CLK_Q},
        {megamol::frontend_resources::Key::KEY_W, CLK_W}, {megamol::frontend_resources::Key::KEY_E, CLK_E},
        {megamol::frontend_resources::Key::KEY_R, CLK_R}, {megamol::frontend_resources::Key::KEY_T, CLK_T},
        {megamol::frontend_resources::Key::KEY_Y, CLK_Y}, {megamol::frontend_resources::Key::KEY_U, CLK_U},
        {megamol::frontend_resources::Key::KEY_I, CLK_I}, {megamol::frontend_resources::Key::KEY_O, CLK_O},
        {megamol::frontend_resources::Key::KEY_P, CLK_P},
        {megamol::frontend_resources::Key::KEY_LEFT_BRACKET, CLK_BracketLeft},
        {megamol::frontend_resources::Key::KEY_CAPS_LOCK, CLK_CapsLock},
        {megamol::frontend_resources::Key::KEY_A, CLK_A}, {megamol::frontend_resources::Key::KEY_S, CLK_S},
        {megamol::frontend_resources::Key::KEY_D, CLK_D}, {megamol::frontend_resources::Key::KEY_F, CLK_F},
        {megamol::frontend_resources::Key::KEY_G, CLK_G}, {megamol::frontend_resources::Key::KEY_H, CLK_H},
        {megamol::frontend_resources::Key::KEY_J, CLK_J}, {megamol::frontend_resources::Key::KEY_K, CLK_K},
        {megamol::frontend_resources::Key::KEY_L, CLK_L},
        {megamol::frontend_resources::Key::KEY_SEMICOLON, CLK_SemicolonAndColon},
        {megamol::frontend_resources::Key::KEY_APOSTROPHE, CLK_ApostropheAndDoubleQuote},
        {megamol::frontend_resources::Key::KEY_LEFT_SHIFT, CLK_LeftShift},
        {megamol::frontend_resources::Key::KEY_BACKSLASH, CLK_NonUsBackslash},
        {megamol::frontend_resources::Key::KEY_Z, CLK_Z}, {megamol::frontend_resources::Key::KEY_X, CLK_X},
        {megamol::frontend_resources::Key::KEY_C, CLK_C}, {megamol::frontend_resources::Key::KEY_V, CLK_V},
        {megamol::frontend_resources::Key::KEY_B, CLK_B}, {megamol::frontend_resources::Key::KEY_N, CLK_N},
        {megamol::frontend_resources::Key::KEY_M, CLK_M},
        {megamol::frontend_resources::Key::KEY_COMMA, CLK_CommaAndLessThan},
        {megamol::frontend_resources::Key::KEY_PERIOD, CLK_PeriodAndBiggerThan},
        {megamol::frontend_resources::Key::KEY_SLASH, CLK_SlashAndQuestionMark},
        {megamol::frontend_resources::Key::KEY_LEFT_CONTROL, CLK_LeftCtrl},
        {megamol::frontend_resources::Key::KEY_LEFT_SUPER, CLK_LeftGui},
        {megamol::frontend_resources::Key::KEY_LEFT_ALT, CLK_LeftAlt},
        //{megamol::frontend_resources::Key::KEY_, CLK_Lang2},
        {megamol::frontend_resources::Key::KEY_SPACE, CLK_Space},
        //{megamol::frontend_resources::Key::KEY_, CLK_Lang1},
        //{megamol::frontend_resources::Key::KEY_, CLK_International2},
        {megamol::frontend_resources::Key::KEY_RIGHT_ALT, CLK_RightAlt},
        {megamol::frontend_resources::Key::KEY_RIGHT_SUPER, CLK_RightGui},
        //{megamol::frontend_resources::Key::KEY_, CLK_Application},
        //{megamol::frontend_resources::Key::KEY_, CLK_LedProgramming},
        //{megamol::frontend_resources::Key::KEY_, CLK_Brightness},
        //{megamol::frontend_resources::Key::KEY_, CLK_F12},
        {megamol::frontend_resources::Key::KEY_PRINT_SCREEN, CLK_PrintScreen},
        {megamol::frontend_resources::Key::KEY_SCROLL_LOCK, CLK_ScrollLock},
        {megamol::frontend_resources::Key::KEY_PAUSE, CLK_PauseBreak},
        {megamol::frontend_resources::Key::KEY_INSERT, CLK_Insert},
        {megamol::frontend_resources::Key::KEY_HOME, CLK_Home},
        {megamol::frontend_resources::Key::KEY_PAGE_UP, CLK_PageUp},
        {megamol::frontend_resources::Key::KEY_RIGHT_BRACKET, CLK_BracketRight},
        {megamol::frontend_resources::Key::KEY_BACKSLASH, CLK_Backslash},
        //{megamol::frontend_resources::Key::KEY_, CLK_NonUsTilde},
        {megamol::frontend_resources::Key::KEY_ENTER, CLK_Enter},
        //{megamol::frontend_resources::Key::KEY_, CLK_International1},
        {megamol::frontend_resources::Key::KEY_EQUAL, CLK_EqualsAndPlus},
        //{megamol::frontend_resources::Key::KEY_, CLK_International3},
        {megamol::frontend_resources::Key::KEY_BACKSPACE, CLK_Backspace},
        {megamol::frontend_resources::Key::KEY_DELETE, CLK_Delete},
        {megamol::frontend_resources::Key::KEY_END, CLK_End},
        {megamol::frontend_resources::Key::KEY_PAGE_DOWN, CLK_PageDown},
        {megamol::frontend_resources::Key::KEY_RIGHT_SHIFT, CLK_RightShift},
        {megamol::frontend_resources::Key::KEY_RIGHT_CONTROL, CLK_RightCtrl},
        {megamol::frontend_resources::Key::KEY_UP, CLK_UpArrow},
        {megamol::frontend_resources::Key::KEY_LEFT, CLK_LeftArrow},
        {megamol::frontend_resources::Key::KEY_DOWN, CLK_DownArrow},
        {megamol::frontend_resources::Key::KEY_RIGHT, CLK_RightArrow},
        //{megamol::frontend_resources::Key::KEY_, CLK_WinLock},
        //{megamol::frontend_resources::Key::KEY_, CLK_Mute},
        //{megamol::frontend_resources::Key::KEY_, CLK_Stop},
        //{megamol::frontend_resources::Key::KEY_, CLK_ScanPreviousTrack},
        //{megamol::frontend_resources::Key::KEY_, CLK_PlayPause},
        //{megamol::frontend_resources::Key::KEY_, CLK_ScanNextTrack},
        {megamol::frontend_resources::Key::KEY_NUM_LOCK, CLK_NumLock},
        {megamol::frontend_resources::Key::KEY_KP_DIVIDE, CLK_KeypadSlash},
        {megamol::frontend_resources::Key::KEY_KP_MULTIPLY, CLK_KeypadAsterisk},
        {megamol::frontend_resources::Key::KEY_KP_SUBTRACT, CLK_KeypadMinus},
        {megamol::frontend_resources::Key::KEY_KP_ADD, CLK_KeypadPlus},
        {megamol::frontend_resources::Key::KEY_KP_ENTER, CLK_KeypadEnter},
        {megamol::frontend_resources::Key::KEY_KP_7, CLK_Keypad7},
        {megamol::frontend_resources::Key::KEY_KP_8, CLK_Keypad8},
        {megamol::frontend_resources::Key::KEY_KP_9, CLK_Keypad9},
        //{megamol::frontend_resources::Key::KEY_KP_DECIMAL, CLK_KeypadComma},
        {megamol::frontend_resources::Key::KEY_KP_4, CLK_Keypad4},
        {megamol::frontend_resources::Key::KEY_KP_5, CLK_Keypad5},
        {megamol::frontend_resources::Key::KEY_KP_6, CLK_Keypad6},
        {megamol::frontend_resources::Key::KEY_KP_1, CLK_Keypad1},
        {megamol::frontend_resources::Key::KEY_KP_2, CLK_Keypad2},
        {megamol::frontend_resources::Key::KEY_KP_3, CLK_Keypad3},
        {megamol::frontend_resources::Key::KEY_KP_0, CLK_Keypad0},
        {megamol::frontend_resources::Key::KEY_KP_DECIMAL, CLK_KeypadPeriodAndDelete},
        //{megamol::frontend_resources::Key::KEY_, CLK_G1},
        //{megamol::frontend_resources::Key::KEY_, CLK_G2},
        //{megamol::frontend_resources::Key::KEY_, CLK_G3},
        //{megamol::frontend_resources::Key::KEY_, CLK_G4},
        //{megamol::frontend_resources::Key::KEY_, CLK_G5},
        //{megamol::frontend_resources::Key::KEY_, CLK_G6},
        //{megamol::frontend_resources::Key::KEY_, CLK_G7},
        //{megamol::frontend_resources::Key::KEY_, CLK_G8},
        //{megamol::frontend_resources::Key::KEY_, CLK_G9},
        //{megamol::frontend_resources::Key::KEY_, CLK_G10},
        //{megamol::frontend_resources::Key::KEY_, CLK_VolumeUp},
        //{megamol::frontend_resources::Key::KEY_, CLK_VolumeDown},
        //{megamol::frontend_resources::Key::KEY_, CLK_MR},
        //{megamol::frontend_resources::Key::KEY_, CLK_M1},
        //{megamol::frontend_resources::Key::KEY_, CLK_M2},
        //{megamol::frontend_resources::Key::KEY_, CLK_M3},
        //{megamol::frontend_resources::Key::KEY_, CLK_G11},
        //{megamol::frontend_resources::Key::KEY_, CLK_G12},
        //{megamol::frontend_resources::Key::KEY_, CLK_G13},
        //{megamol::frontend_resources::Key::KEY_, CLK_G14},
        //{megamol::frontend_resources::Key::KEY_, CLK_G15},
        //{megamol::frontend_resources::Key::KEY_, CLK_G16},
        //{megamol::frontend_resources::Key::KEY_, CLK_G17},
        //{megamol::frontend_resources::Key::KEY_, CLK_G18},
        //{megamol::frontend_resources::Key::KEY_, CLK_International5},
        //{megamol::frontend_resources::Key::KEY_, CLK_International4},
    };

const char* corsair_error_to_string(CorsairError error) {
    switch (error) {
    case CE_Success:
        return "CE_Success";
    case CE_ServerNotFound:
        return "CE_ServerNotFound";
    case CE_NoControl:
        return "CE_NoControl";
    case CE_ProtocolHandshakeMissing:
        return "CE_ProtocolHandshakeMissing";
    case CE_IncompatibleProtocol:
        return "CE_IncompatibleProtocol";
    case CE_InvalidArguments:
        return "CE_InvalidArguments";
    default:
        return "unknown error";
    }
}
#endif


megamol::frontend_resources::CommandRegistry::CommandRegistry() {
#ifdef MEGAMOL_USE_CUESDK
    CorsairPerformProtocolHandshake();
    if (const auto error = CorsairGetLastError()) {
        std::cout << "Corsair CUE Handshake failed: " << corsair_error_to_string(error) << " - is iCUE running?"
                  << std::endl;
    }
    CorsairRequestControl(CAM_ExclusiveLightingControl);
    led_positions = CorsairGetLedPositions();
    for (auto i = 0; i < led_positions->numberOfLed; i++) {
        const auto ledPos = led_positions->pLedPosition[i];
        auto ledColor = CorsairLedColor{ledPos.ledId, 0, 0, 0};
        if (ledPos.ledId == CLK_LeftShift || ledPos.ledId == CLK_RightShift || ledPos.ledId == CLK_LeftCtrl ||
            ledPos.ledId == CLK_RightCtrl || ledPos.ledId == CLK_LeftAlt || ledPos.ledId == CLK_RightAlt) {
            ledColor.b = 255;
        }
        black_keyboard.push_back(ledColor);
    }
    if (!CorsairSetLedsColors(static_cast<int>(black_keyboard.size()), black_keyboard.data())) {
        const auto error = CorsairGetLastError();
        std::cout << "Setting Corsair leds to default failed: " << corsair_error_to_string(error) << std::endl;
    }
#endif
}

megamol::frontend_resources::CommandRegistry::~CommandRegistry() {
#ifdef MEGAMOL_USE_CUESDK
    CorsairReleaseControl(CAM_ExclusiveLightingControl);
#endif
}

std::string megamol::frontend_resources::CommandRegistry::add_command(const megamol::frontend_resources::Command& c) {
    const bool command_is_new = is_new(c.name);
    const bool key_code_unused = key_to_command.find(c.key) == key_to_command.end();
    if (command_is_new && key_code_unused) {
        push_command(c);
        return c.name;
    } else {
        Command c2;
        if (!command_is_new) {
            c2.name = increment_name(c.name);
        } else {
            c2.name = c.name;
        }
        if (key_code_unused) {
            c2.key = c.key;
        }
        c2.parent = c.parent;
        c2.parent_type = c.parent_type;
        c2.effect = c.effect;
        push_command(c2);
        return c2.name;
    }
}

const megamol::frontend_resources::Command megamol::frontend_resources::CommandRegistry::get_command(
    const KeyCode& key) const {
    const auto c = key_to_command.find(key);
    if (c == key_to_command.end()) {
        return Command{};
    } else {
        return commands[c->second];
    }
}

const megamol::frontend_resources::Command megamol::frontend_resources::CommandRegistry::get_command(
    const std::string& command_name) {
    if (is_new(command_name)) {
        return Command{};
    } else {
        auto idx = command_index[command_name];
        return commands[idx];
    }
}

void megamol::frontend_resources::CommandRegistry::remove_command_by_parent(const std::string& parent_param) {
    auto it = std::find_if(
        commands.begin(), commands.end(), [parent_param](const Command& c) { return c.parent == parent_param; });
    if (it != commands.end()) {
        if (it->key.key != Key::KEY_UNKNOWN) {
            remove_color_from_layer(*it);
            key_to_command.erase(it->key);
        }
        commands.erase(it);
        rebuild_index();
    }
}

void megamol::frontend_resources::CommandRegistry::remove_command_by_name(const std::string& command_name) {
    if (!is_new(command_name)) {
        auto idx = command_index[command_name];
        auto& c = commands[idx];
        if (c.key.key != Key::KEY_UNKNOWN) {
            remove_color_from_layer(c);
            key_to_command.erase(c.key);
        }
        commands.erase(commands.begin() + idx);
        rebuild_index();
    }
}

bool megamol::frontend_resources::CommandRegistry::update_hotkey(const std::string& command_name, KeyCode key) {
    if (!is_new(command_name)) {
        auto& c = commands[command_index[command_name]];
        const auto old_key = c.key;
        if (old_key.key != Key::KEY_UNKNOWN) {
            remove_color_from_layer(c);
            key_to_command.erase(old_key);
        }
        c.key = key;
        if (key.key != Key::KEY_UNKNOWN) {
            // important! deserialization might result in different "stealing" order of duplicate hotkeys
            // these will be fixed by subsequent hotkey updates, but might result in temporary duplication during deserialization
            remove_hotkey(key);
            key_to_command[key] = command_index[command_name];
            add_color_to_layer(c);
        }
        modifiers_changed(current_modifiers);
        return true;
    }
    return false;
}

bool megamol::frontend_resources::CommandRegistry::remove_hotkey(KeyCode key) {
    auto it = key_to_command.find(key);
    if (it != key_to_command.end()) {
        update_hotkey(commands[it->second].name, Key::KEY_UNKNOWN);
        return true;
    }
    return false;
}

void megamol::frontend_resources::CommandRegistry::modifiers_changed(Modifiers mod) {
    // TODO
#ifdef MEGAMOL_USE_CUESDK
    // unset all keys
    if (!CorsairSetLedsColors(static_cast<int>(black_keyboard.size()), black_keyboard.data())) {
        const auto error = CorsairGetLastError();
        std::cout << "Setting Corsair leds to default failed: " << corsair_error_to_string(error) << std::endl;
    }
    // find layer that matches the modifier
    const auto layer = key_colors.find(mod);
    if (layer != key_colors.end()) {
        if (!CorsairSetLedsColors(static_cast<int>(layer->second.size()), layer->second.data())) {
            const auto error = CorsairGetLastError();
            std::cout << "Setting Corsair leds for layer " << mod.ToString()
                      << " failed: " << corsair_error_to_string(error) << std::endl;
        }
    }
#endif
    current_modifiers = mod;
}

bool megamol::frontend_resources::CommandRegistry::exec_command(const std::string& command_name) const {
    const auto& it = command_index.find(command_name);
    if (it != command_index.end()) {
        commands[it->second].execute();
        return true;
    } else {
        return false;
    }
}

bool megamol::frontend_resources::CommandRegistry::exec_command(const KeyCode& key) const {
    const auto c = key_to_command.find(key);
    if (c == key_to_command.end()) {
        return false;
    } else {
        commands[c->second].execute();
        return true;
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

void megamol::frontend_resources::CommandRegistry::add_color_to_layer(const megamol::frontend_resources::Command& c) {
#ifdef MEGAMOL_USE_CUESDK
    auto cols = key_colors.find(c.key.mods);
    if (cols == key_colors.end()) {
        key_colors[c.key.mods] = std::vector<CorsairLedColor>();
    }
    if (c.key.key != Key::KEY_UNKNOWN) {
        CorsairLedColor clc{corsair_led_from_glfw_key[c.key.key], 255, 0, 0};
        auto& layer = key_colors[c.key.mods];
        auto it = std::find_if(
            layer.begin(), layer.end(), [&](const CorsairLedColor& cled) { return cled.ledId == clc.ledId; });
        if (it == layer.end()) {
            key_colors[c.key.mods].push_back(clc);
        } else {
            std::cout << "Warning: Corsair LEDs inconsistent: " << clc.ledId << " (from " << c.key.ToString()
                      << ") already used" << std::endl;
        }
    }
    std::cout << "adding " << c.name << " with " << c.key.ToString() << std::endl;
#endif
}

void megamol::frontend_resources::CommandRegistry::remove_color_from_layer(
    const megamol::frontend_resources::Command& c) {
#ifdef MEGAMOL_USE_CUESDK
    if (c.key.key != Key::KEY_UNKNOWN) {
        auto& layer = key_colors[c.key.mods];
        const auto& k = corsair_led_from_glfw_key[c.key.key];
        const auto it =
            std::remove_if(layer.begin(), layer.end(), [k](const CorsairLedColor& c) { return c.ledId == k; });
        layer.erase(it);
    }
#endif
}

void megamol::frontend_resources::CommandRegistry::push_command(const Command& c) {
    commands.push_back(c);
    if (c.key.key != Key::KEY_UNKNOWN) {
        key_to_command[c.key] = static_cast<int>(commands.size() - 1);
        add_color_to_layer(c);
#ifdef MEGAMOL_USE_CUESDK
        if (current_modifiers.equals(c.key.mods)) {
            auto ledColor = CorsairLedColor{corsair_led_from_glfw_key[c.key.key], 255, 0, 0};
            CorsairSetLedsColors(1, &ledColor);
        }
#endif
    }
    command_index[c.name] = static_cast<int>(commands.size() - 1);
}

void megamol::frontend_resources::CommandRegistry::rebuild_index() {
    command_index.clear();
    key_to_command.clear();
    for (auto x = 0; x < commands.size(); ++x) {
        auto& c = commands[x];
        command_index[commands[x].name] = x;
        if (c.key.key != Key::KEY_UNKNOWN) {
            key_to_command[c.key] = x;
        }
    }
}
