/*
 * PopUp.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GUIUtils.h"

#include "vislib/UTF8Encoder.h"

#include <imgui_stdlib.h>
#include <vector>


using namespace megamol::gui;


GUIUtils::GUIUtils(void)
    : tooltip_time(0.0f)
    , tooltip_id(GUI_INVALID_ID)
    , search_focus(false)
    , search_string()
    , file_name_str()
    , file_path_str()
    , additional_lines(0) {}


void GUIUtils::HoverToolTip(const std::string& text, ImGuiID id, float time_start, float time_end) {
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


void GUIUtils::HelpMarkerToolTip(const std::string& text, std::string label) {
    assert(ImGui::GetCurrentContext() != nullptr);

    if (!text.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled(label.c_str());
        this->HoverToolTip(text);
    }
}


float GUIUtils::TextWidgetWidth(const std::string& text) const {
    assert(ImGui::GetCurrentContext() != nullptr);

    ImVec2 pos = ImGui::GetCursorPos();
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.0f);
    ImGui::Text(text.c_str());
    ImGui::PopStyleVar();
    ImGui::SetCursorPos(pos);

    return ImGui::GetItemRectSize().x;
}


bool GUIUtils::Utf8Decode(std::string& str) const {
    vislib::StringA dec_tmp;
    if (vislib::UTF8Encoder::Decode(dec_tmp, vislib::StringA(str.c_str()))) {
        str = std::string(dec_tmp.PeekBuffer());
        return true;
    }
    return false;
}


bool GUIUtils::Utf8Encode(std::string& str) const {
    vislib::StringA dec_tmp;
    if (vislib::UTF8Encoder::Encode(dec_tmp, vislib::StringA(str.c_str()))) {
        str = std::string(dec_tmp.PeekBuffer());
        return true;
    }
    return false;
}

void megamol::gui::GUIUtils::StringSearch(const std::string& id, const std::string& help) {
    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    ImGui::BeginGroup();
    ImGui::PushID(id.c_str());

    if (ImGui::Button("Clear")) {
        this->search_string = "";
    }
    ImGui::SameLine();

    // Set keyboard focus when hotkey is pressed
    if (this->search_focus) {
        ImGui::SetKeyboardFocusHere();
        this->search_focus = false;
    }

    std::string complete_label = "Search (?)";
    auto width = ImGui::GetContentRegionAvailWidth() - ImGui::GetCursorPosX() + 4.0f * style.ItemInnerSpacing.x -
                 this->TextWidgetWidth(complete_label);
    const int min_width = 50.0f;
    width = (width < min_width) ? (min_width) : width;
    ImGui::PushItemWidth(width);

    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
    this->Utf8Encode(this->search_string);
    ImGui::InputText("Search", &this->search_string, ImGuiInputTextFlags_AutoSelectAll);
    this->Utf8Decode(this->search_string);

    ImGui::PopItemWidth();
    ImGui::SameLine();

    this->HelpMarkerToolTip(help);

    ImGui::PopID();
    ImGui::EndGroup();
}


bool megamol::gui::GUIUtils::VerticalSplitter(float* size_left, float* size_right) {
    assert(ImGui::GetCurrentContext() != nullptr);

    const float thickness = 12.0f;

    bool split_vertically = true;
    float min_size = 1.0f; // >=1.0!
    float splitter_long_axis_size = ImGui::GetContentRegionAvail().y;

    float width_avail = ImGui::GetWindowSize().x - (3.0f * thickness);

    if (width_avail < thickness) return false;

    (*size_left) = std::min((*size_left), width_avail);
    (*size_right) = width_avail - (*size_left);

    ImGuiContext& g = *GImGui;
    ImGuiWindow* window = g.CurrentWindow;
    ImGuiID id = window->GetID("##Splitter");
    ImRect bb;
    bb.Min = window->DC.CursorPos +
             (split_vertically ? ImVec2((*size_left) + 1.0f, 0.0f) : ImVec2(0.0f, (*size_left) + 1.0f));
    bb.Max = bb.Min + ImGui::CalcItemSize(split_vertically ? ImVec2(thickness - 4.0f, splitter_long_axis_size)
                                                           : ImVec2(splitter_long_axis_size, thickness - 4.0f),
                          0.0f, 0.0f);
    return ImGui::SplitterBehavior(
        bb, id, split_vertically ? ImGuiAxis_X : ImGuiAxis_Y, size_left, size_right, min_size, min_size, 0.0f);
}

#ifdef GUI_USE_FILESYSTEM
bool megamol::gui::GUIUtils::FileBrowserPopUp(
    FileBrowserFlag flag, bool open_popup, const std::string& label, std::string& inout_filename) {

    bool retval = false;

    try {
        std::string popup_name = label;
        ImGui::PushID(label.c_str());

        if (open_popup) {
            // Check given file name path
            fsns::path file_path = static_cast<fsns::path>(inout_filename);
            if (file_path.empty() || !fsns::exists(file_path)) {
                file_path = fsns::current_path();
            }
            this->splitPath(file_path, this->file_path_str, this->file_name_str);

            ImGui::OpenPopup(popup_name.c_str());
            // Set initial window size of pop up
            ImGui::SetNextWindowSize(ImVec2(600.0f, 300.0f));
        }

        bool open = true;
        if (ImGui::BeginPopupModal(popup_name.c_str(), &open, ImGuiWindowFlags_None)) {

            fsns::path file_path;

            const std::string ext = ".lua";
            const auto error_color = ImVec4(0.9f, 0.0f, 0.0f, 1.0f);
            const auto warning_color = ImVec4(0.75f, 0.75f, 0.f, 1.0f);
            std::string error;
            std::string warning;

            ImGuiStyle& style = ImGui::GetStyle();

            // Path ---------------------------------------------------
            bool valid_directory = false;
            /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
            this->Utf8Encode(this->file_path_str);
            ImGui::InputText("Directory", &this->file_path_str, ImGuiInputTextFlags_AutoSelectAll);
            this->Utf8Decode(this->file_path_str);
            // Validating directory
            try {
                file_path = static_cast<fsns::path>(this->file_path_str);
                if (fsns::status_known(fsns::status(file_path)) && fsns::is_directory(file_path)) {
                    valid_directory = true;
                }
            } catch (fsns::filesystem_error e) {
                // vislib::sys::Log::DefaultLog.WriteError(
                //    "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
            } catch (std::exception e) {
                // vislib::sys::Log::DefaultLog.WriteError(
                //    "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
            }
            if (!valid_directory) {
                // Error when path is no valid directory
                error = "Invalid Directory";
                ImGui::TextColored(error_color, error.c_str());
            }

            // File browser selectables ---------------------------------------

            auto select_flags = ImGuiSelectableFlags_DontClosePopups;
            float child_select_height =
                (ImGui::GetContentRegionAvail().y - (ImGui::GetTextLineHeightWithSpacing() * this->additional_lines) -
                    ImGui::GetItemsLineHeightWithSpacing() * 2.0f);
            this->additional_lines = 0;

            // Files and directories ----------------
            ImGui::BeginChild(
                "files_list_child_window", ImVec2(0.0f, child_select_height), true, ImGuiWindowFlags_None);

            if (valid_directory) {

                // Parent directory selectable
                std::string tag_parent = "..";
                if (ImGui::Selectable(tag_parent.c_str(), false, select_flags)) {
                    if (file_path.has_parent_path()) {
                        this->file_path_str = file_path.parent_path().generic_u8string();
                    }
                }

                std::vector<fsns::path> paths;
                std::copy(fsns::directory_iterator(file_path), fsns::directory_iterator(), std::back_inserter(paths));

                // Sort path case insensitive alphabetically ascending
                std::sort(paths.begin(), paths.end(), [](fsns::path const& a, fsns::path const& b) {
                    std::string a_str = a.filename().generic_u8string();
                    for (auto& c : a_str) c = std::toupper(c);
                    std::string b_str = b.filename().generic_u8string();
                    for (auto& c : b_str) c = std::toupper(c);
                    return (a_str < b_str);
                });

                // Files and directories of current directory
                for (const auto& path : paths) {
                    // Different color for directories
                    if (fsns::is_directory(path)) {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                    }
                    auto label = path.filename().generic_u8string();
                    if (ImGui::Selectable(label.c_str(), (label == this->file_name_str), select_flags)) {
                        this->splitPath(path, this->file_path_str, this->file_name_str);
                    }
                    if (fsns::is_directory(path)) {
                        ImGui::PopStyleColor();
                    }
                }
            }

            ImGui::EndChild();

            /// Errors and warnings ...


            // Widget group -------------------------------------------------------
            ImGui::BeginGroup();

            bool apply = false;

            // File name ------------------------
            bool valid_file = true;
            bool valid_ending = true;
            if (flag == GUIUtils::FileBrowserFlag::SAVE) {
                /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
                this->Utf8Encode(this->file_name_str);
                if (ImGui::InputText("File Name", &this->file_name_str,
                        ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll)) {
                    apply = true;
                }
                this->Utf8Decode(this->file_name_str);

                // Validating file name
                if (this->file_name_str.empty()) {
                    error = "Enter file name.";
                    ImGui::TextColored(error_color, error.c_str());
                    this->additional_lines++;
                    valid_file = false;
                } else {
                    if (!file::FileExtension<std::string>(this->file_name_str, ext)) {
                        // Warn when file has not required extension
                        warning = "Appending required file extension '" + ext + "'";
                        ImGui::TextColored(warning_color, warning.c_str());
                        this->additional_lines++;
                        valid_ending = false;
                    }
                    std::string actual_filename = this->file_name_str;
                    if (!valid_ending) {
                        actual_filename.append(ext);
                    }
                    file_path = static_cast<fsns::path>(this->file_path_str) / static_cast<fsns::path>(actual_filename);
                    if (fsns::exists(file_path) && fsns::is_regular_file(file_path)) {
                        // Warn when file already exists
                        warning = "Overwriting existing file.";
                        ImGui::TextColored(warning_color, warning.c_str());
                        this->additional_lines++;
                    }
                    if (fsns::is_directory(file_path)) {
                        // Error when file is directory
                        error = "File is directory.";
                        ImGui::TextColored(error_color, error.c_str());
                        this->additional_lines++;
                        valid_file = false;
                    }
                }
            } else if (flag == GUIUtils::FileBrowserFlag::LOAD) {
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
                this->Utf8Encode(this->file_name_str);
                ImGui::InputText("File Name", &this->file_name_str);
                this->Utf8Decode(this->file_name_str);
                ImGui::PopItemFlag();

                // Validating selected name
                if (!file::FileExtension<std::string>(this->file_name_str, ext)) {
                    // Error when file has not required extension
                    error = "File with extension '" + ext + "' required.";
                    ImGui::TextColored(error_color, error.c_str());
                    this->additional_lines++;
                    valid_ending = false;
                    valid_file = false;
                }
            }

            // Buttons ------------------------------
            // float cancel_button_width = this->TextWidgetWidth("Cancel") + style.ItemSpacing.x;
            std::string label;
            if (flag == GUIUtils::FileBrowserFlag::SAVE) {
                label = "Save";
            } else if (flag == GUIUtils::FileBrowserFlag::LOAD) {
                label = "Load";
            }
            const auto err_btn_color = ImVec4(0.6f, 0.0f, 0.0f, 1.0f);
            const auto er_btn_hov_color = ImVec4(0.9f, 0.0f, 0.0f, 1.0f);
            ImGui::PushStyleColor(
                ImGuiCol_Button, !(valid_directory && valid_file) ? err_btn_color : style.Colors[ImGuiCol_Button]);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                !(valid_directory && valid_file) ? er_btn_hov_color : style.Colors[ImGuiCol_ButtonHovered]);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, style.Colors[ImGuiCol_ButtonActive]);
            if (ImGui::Button(label.c_str())) {
                apply = true;
            }
            ImGui::PopStyleColor(3);

            ImGui::SameLine();
            // ImGui::SameLine(ImGui::GetContentRegionAvailWidth() - cancel_button_width);
            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
            }

            // Return file path -----------------------------------------------
            if (apply && valid_directory && valid_file) {
                // Appending required extension
                if (!valid_ending) {
                    this->file_name_str.append(ext);
                }
                file_path = static_cast<fsns::path>(this->file_path_str) / static_cast<fsns::path>(this->file_name_str);
                inout_filename = file_path.generic_u8string();
                ImGui::CloseCurrentPopup();
                retval = true;
            }

            ImGui::EndGroup();
            // --------------------------------------------------------------------

            ImGui::EndPopup();
        }

        ImGui::PopID();

    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return retval;
}


bool megamol::gui::GUIUtils::splitPath(const fsns::path& in_file_path, std::string& out_path, std::string& out_file) {

    if (fsns::is_regular_file(in_file_path)) {
        if (in_file_path.has_parent_path()) {
            out_path = in_file_path.parent_path().generic_u8string();
        }
        if (in_file_path.has_filename()) {
            out_file = in_file_path.filename().generic_u8string();
        }
    } else {
        out_path = in_file_path.generic_u8string();
        out_file.clear();
    }
    return true;
}


#endif // GUI_USE_FILESYSTEM