/*
 * LogConsole.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "stdafx.h"
#include "LogConsole.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::LogConsole::LogConsole()
        : echo_log_buffer()
        , echo_log_stream(&this->echo_log_buffer)
        , echo_log_target(nullptr)
        , log()
        , log_level(megamol::core::utility::log::Log::LEVEL_ALL)
        , scroll_log_down(2)
        , scroll_log_up(0)
        , last_window_height(0.0f)
        , tooltip() {

    this->echo_log_target =
        std::make_shared<megamol::core::utility::log::StreamTarget>(this->echo_log_stream, this->log_level);

    this->connect_log();
}


LogConsole::~LogConsole() {

    // Reset echo target only if log target of this class instance is used
    if (megamol::core::utility::log::Log::DefaultLog.AccessEchoTarget() == this->echo_log_target) {
        megamol::core::utility::log::Log::DefaultLog.SetEchoTarget(nullptr);
    }
    this->echo_log_target.reset();
}


bool megamol::gui::LogConsole::Draw(WindowCollection::WindowConfiguration& wc) {

    this->log_level = wc.log_level;

    // Scroll down if window height changes
    if (this->last_window_height != ImGui::GetWindowHeight()) {
        this->last_window_height = ImGui::GetWindowHeight();
        this->scroll_log_down = 2;
    }

    // Menu
    if (ImGui::BeginMenuBar()) {

        // Force Open on Warnings and Errors
        ImGui::Checkbox("Force Open", &wc.log_force_open);
        this->tooltip.Marker("Force open log console window on warnings and errors.");
        ImGui::Separator();

        // Log Level
        ImGui::TextUnformatted("Show");
        ImGui::SameLine();
        if (ImGui::RadioButton("Errors", (this->log_level >= megamol::core::utility::log::Log::LEVEL_ERROR))) {
            if (this->log_level >= megamol::core::utility::log::Log::LEVEL_ERROR) {
                this->log_level = megamol::core::utility::log::Log::LEVEL_NONE;
            } else {
                this->log_level = megamol::core::utility::log::Log::LEVEL_ERROR;
            }
            this->scroll_log_down = 2;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Warnings", (this->log_level >= megamol::core::utility::log::Log::LEVEL_WARN))) {
            if (this->log_level >= megamol::core::utility::log::Log::LEVEL_WARN) {
                this->log_level = megamol::core::utility::log::Log::LEVEL_ERROR;
            } else {
                this->log_level = megamol::core::utility::log::Log::LEVEL_WARN;
            }
            this->scroll_log_down = 2;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Infos", (this->log_level == megamol::core::utility::log::Log::LEVEL_ALL))) {
            if (this->log_level == megamol::core::utility::log::Log::LEVEL_ALL) {
                this->log_level = megamol::core::utility::log::Log::LEVEL_WARN;
            } else {
                this->log_level = megamol::core::utility::log::Log::LEVEL_ALL;
            }
            this->scroll_log_down = 2;
        }

        // Scrolling
        std::string scroll_label = "Scroll";
        ImGui::SameLine(0.0f, ImGui::GetContentRegionAvail().x - (2.25f * ImGui::GetFrameHeightWithSpacing()) -
                                  ImGui::CalcTextSize(scroll_label.c_str()).x);
        ImGui::TextUnformatted(scroll_label.c_str());
        ImGui::SameLine();
        if (ImGui::ArrowButton("scroll_up", ImGuiDir_Up)) {
            this->scroll_log_up = 2;
        }
        this->tooltip.ToolTip("Scroll to first log entry.");
        ImGui::SameLine();
        if (ImGui::ArrowButton("scroll_down", ImGuiDir_Down)) {
            this->scroll_log_down = 2;
        }
        this->tooltip.ToolTip("Scroll to last log entry.");

        ImGui::EndMenuBar();
    }

    // Scroll (requires 2 frames for being applyed)
    if (this->scroll_log_down > 0) {
        ImGui::SetScrollY(ImGui::GetScrollMaxY());
        this->scroll_log_down--;
    }
    if (this->scroll_log_up > 0) {
        ImGui::SetScrollY(0);
        this->scroll_log_up--;
    }

    // Print messages
    for (auto& entry : this->log) {
        if (entry.level <= this->log_level) {
            if (entry.level >= megamol::core::utility::log::Log::LEVEL_INFO) {
                ImGui::TextUnformatted(entry.message.c_str());
            } else if (entry.level >= megamol::core::utility::log::Log::LEVEL_WARN) {
                ImGui::TextColored(GUI_COLOR_TEXT_WARN, entry.message.c_str());
            } else if (entry.level >= megamol::core::utility::log::Log::LEVEL_ERROR) {
                ImGui::TextColored(GUI_COLOR_TEXT_ERROR, entry.message.c_str());
            }
        }
    }

    wc.log_level = this->log_level;

    return true;
}


bool megamol::gui::LogConsole::Update(WindowCollection::WindowConfiguration& wc) {

    // Get new messages
    bool updated = false;
    std::vector<std::string> new_messages;
    if (this->echo_log_buffer.ConsumeMessage(new_messages)) {
        for (auto& msg : new_messages) {
            // Assuming one log message of format "<level>|<message>\r\n"
            auto seperator_index = msg.find("|");
            if (seperator_index != std::string::npos) {
                unsigned int new_log_level = megamol::core::utility::log::Log::LEVEL_NONE;
                auto level_str = msg.substr(0, seperator_index);
                try {
                    new_log_level = std::stoi(level_str);
                } catch (...) {}
                if (new_log_level != megamol::core::utility::log::Log::LEVEL_NONE) {
                    LogEntry new_entry;
                    new_entry.level = new_log_level;
                    new_entry.message = msg;
                    this->log.push_back(new_entry);

                    // Force open log window if there is any warning
                    if (wc.log_force_open && (new_log_level < megamol::core::utility::log::Log::LEVEL_INFO)) {
                        this->log_level = megamol::core::utility::log::Log::LEVEL_WARN;
                        wc.win_show = true;
                    }

                    this->scroll_log_down = 2;
                    updated = true;
                }
            }
        }
    }
    return updated;
}


bool megamol::gui::LogConsole::connect_log(void) {

    auto current_echo_target = megamol::core::utility::log::Log::DefaultLog.AccessEchoTarget();
    std::shared_ptr<megamol::core::utility::log::OfflineTarget> offline_echo_target =
        std::dynamic_pointer_cast<megamol::core::utility::log::OfflineTarget>(current_echo_target);

    // Only connect if echo target is still default OfflineTarget
    /// Note: A second log console is temporarily created when "GUIView" module is loaded in configurator for complete
    /// module list. For this "GUIView" module NO log is connected, because the main LogConsole instance is already
    /// connected and the taget is not the default OfflineTarget.
    if ((offline_echo_target != nullptr) && (this->echo_log_target != nullptr)) {
        megamol::core::utility::log::Log::DefaultLog.SetEchoTarget(this->echo_log_target);
    }

    return true;
}
