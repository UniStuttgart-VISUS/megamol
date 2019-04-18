/*
 * GUIUtility.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GUIUtility.h"


using namespace megamol::gui;


/**
 * Ctor
 */
megamol::gui::GUIUtility::GUIUtility(void) {

    // nothing to do here ...
}


/**
 * Dtor
 */
megamol::gui::GUIUtility::~GUIUtility(void) {

    // nothing to do here ...
}


/**
 * GUIUtility::FilePathExists
 */
bool megamol::gui::GUIUtility::FilePathExists(PathType path) { return ns_fs::exists(path); }

bool megamol::gui::GUIUtility::FilePathExists(std::string path) { return this->FilePathExists(PathType(path)); }

bool megamol::gui::GUIUtility::FilePathExists(std::wstring path) { return this->FilePathExists(PathType(path)); }


/**
 * GUIUtility::FileHasExtension
 */
bool megamol::gui::GUIUtility::FileHasExtension(PathType path, std::string ext) {

    if (!this->FilePathExists(path)) {
        return false;
    }

    return (path.extension().generic_string() == ext);
}

bool megamol::gui::GUIUtility::FileHasExtension(std::string path, std::string ext) {
    return this->FileHasExtension(PathType(path), ext);
}

bool megamol::gui::GUIUtility::FileHasExtension(std::wstring path, std::string ext) {
    return this->FileHasExtension(PathType(path), ext);
}


/**
 * GUIUtility::SearchFilePathRecursive
 */
std::string megamol::gui::GUIUtility::SearchFilePathRecursive(std::string file, PathType search_path) {

    std::string found_file_path;

    for (auto& entry : ns_fs::recursive_directory_iterator(search_path)) {
        if (entry.path().filename().generic_string() == file) {
            found_file_path = entry.path().generic_string();
            break;
        }
    }
    return found_file_path;
}

std::string megamol::gui::GUIUtility::SearchFilePathRecursive(std::string file, std::string search_path) {
    return this->SearchFilePathRecursive(file, PathType(search_path));
}

std::string megamol::gui::GUIUtility::SearchFilePathRecursive(std::string file, std::wstring search_path) {
    return this->SearchFilePathRecursive(file, PathType(search_path));
}


/**
 * GUIUtility::HoverToolTip
 */
void megamol::gui::GUIUtility::HoverToolTip(std::string text, ImGuiID id, float time_start, float time_end) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiIO& io = ImGui::GetIO();

    if (ImGui::IsItemHovered()) {
        bool show_tooltip = false;
        if (time_start > 0.0f) {
            if (this->tooltip_id != id) {
                this->tooltip_time = 0.0f;
                this->tooltip_id = id;
            } else {
                if ((this->tooltip_time > time_start) && (this->tooltip_time < (time_start + time_end))) {
                    show_tooltip = true;
                }
                this->tooltip_time += io.DeltaTime;
            }
        } else {
            show_tooltip = true;
        }

        if (show_tooltip) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(text.c_str());
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    } else {
        if ((time_start > 0.0f) && (this->tooltip_id == id)) {
            this->tooltip_time = 0.0f;
        }
    }
}


/**
 * GUIUtility::HelpMarkerToolTip
 */
void megamol::gui::GUIUtility::HelpMarkerToolTip(std::string text, std::string label) {

    assert(ImGui::GetCurrentContext() != nullptr);

    if (!text.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled(label.c_str());
        this->HoverToolTip(text);
    }
}


/**
 * GUIUtility::InputDialogPopUp
 */
std::string megamol::gui::GUIUtility::InputDialogPopUp(std::string popup_name, std::string request, bool open) {

    assert(ImGui::GetCurrentContext() != nullptr);

    std::string outtext;

    size_t bufferLength = GUI_MAX_BUFFER_LEN;
    char* buffer = new char[bufferLength];
    buffer[0] = '\0';

    if (open) {
        ImGui::OpenPopup(popup_name.c_str());
    }
    if (ImGui::BeginPopupModal(popup_name.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {

        ImGui::Text("Enter %s:", request.c_str());
        this->HelpMarkerToolTip("Press [Enter] to confirm input.");

        // ImGui::SetKeyboardFocusHere();
        if (ImGui::InputText("", buffer, bufferLength, ImGuiInputTextFlags_EnterReturnsTrue)) {
            outtext = buffer;
            ImGui::CloseCurrentPopup();
        }

        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    delete[] buffer;

    return outtext;
}
