/*
 * FileBrowserWidget.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "FileBrowserWidget.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::FileBrowserWidget::FileBrowserWidget(void)
#ifdef GUI_USE_FILESYSTEM
    : search_widget()
    , file_name_str()
    , file_path_str()
    , path_changed(false)
    , valid_directory(false)
    , valid_file(false)
    , valid_ending(false)
    , file_error()
    , file_warning()
    , child_paths()
    , additional_lines(0) {
#else
{
    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
        "[GUI] Filesystem functionality is not available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
#endif // GUI_USE_FILESYSTEM
}


bool megamol::gui::FileBrowserWidget::PopUp(megamol::gui::FileBrowserWidget::FileBrowserFlag flag,
    const std::string& label_id, bool open_popup, std::string& inout_filename) {

    bool retval = false;

    try {
        ImGui::PushID(label_id.c_str());

        if (open_popup) {
#ifdef GUI_USE_FILESYSTEM
            // Check given file name path
            stdfs::path tmp_file_path = static_cast<stdfs::path>(inout_filename);
            if (tmp_file_path.empty() || !stdfs::exists(tmp_file_path)) {
                tmp_file_path = stdfs::current_path();
            }
            this->splitPath(tmp_file_path, this->file_path_str, this->file_name_str);
            this->validateDirectory(this->file_path_str);
            this->validateFile(this->file_name_str, flag);
            this->path_changed = true;

            ImGui::OpenPopup(label_id.c_str());
            // Set initial window size of pop up
            ImGui::SetNextWindowSize(ImVec2(400.0f, 500.0f));
#else
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] Filesystem functionality is not available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
#endif // GUI_USE_FILESYSTEM
        }

#ifdef GUI_USE_FILESYSTEM
        bool open = true;
        if (ImGui::BeginPopupModal(label_id.c_str(), &open, ImGuiWindowFlags_None)) {

            bool apply = false;

            // Path ---------------------------------------------------
            auto last_file_path_str = this->file_path_str;
            if (ImGui::ArrowButton("###arrow_up_directory", ImGuiDir_Up)) {
                stdfs::path tmp_file_path = static_cast<stdfs::path>(this->file_path_str);
                if (tmp_file_path.has_parent_path()) {
                    // Assuming that parent is still valid directory
                    this->file_path_str = tmp_file_path.parent_path().generic_u8string();
                }
            }
            ImGui::SameLine();
            /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
            GUIUtils::Utf8Encode(this->file_path_str);
            ImGui::InputText("Directory", &this->file_path_str, ImGuiInputTextFlags_AutoSelectAll);
            GUIUtils::Utf8Decode(this->file_path_str);
            if (last_file_path_str != this->file_path_str) {
                this->path_changed = true;
                this->validateDirectory(this->file_path_str);
            }
            // Error message when path is no valid directory
            if (!this->valid_directory) {
                ImGui::TextColored(GUI_COLOR_TEXT_ERROR, "Invalid Directory");
            }

            // Search -------------------------------
            std::string help_test = "Case insensitive substring search in\nlisted file and directory names.";
            this->search_widget.Widget("guiwindow_parameter_earch", help_test);
            auto currentSearchString = this->search_widget.GetSearchString();

            // File browser selectables ---------------------------------------
            auto select_flags = ImGuiSelectableFlags_DontClosePopups;
            float child_select_height =
                (ImGui::GetContentRegionAvail().y - (ImGui::GetTextLineHeightWithSpacing() * this->additional_lines) -
                    ImGui::GetFrameHeightWithSpacing() * 2.0f);
            ImGui::BeginChild(
                "files_list_child_window", ImVec2(0.0f, child_select_height), true, ImGuiWindowFlags_None);

            if (this->valid_directory) {
                // Parent directory selectable
                std::string tag_parent("..");
                if (ImGui::Selectable(tag_parent.c_str(), false, select_flags)) {
                    stdfs::path tmp_file_path = static_cast<stdfs::path>(this->file_path_str);
                    if (tmp_file_path.has_parent_path()) {
                        // Assuming that parent is still valid directory
                        this->file_path_str = tmp_file_path.parent_path().generic_u8string();
                        this->path_changed = true;
                    }
                }

                // Only update child paths when path changed.
                if (this->path_changed) {
                    // Reset scrolling
                    ImGui::SetScrollY(0.0f);

                    // Convert realtive path to absolute path
                    if (this->file_path_str == "..") {
                        auto absolute_path = stdfs::absolute(static_cast<stdfs::path>(this->file_path_str));
                        if (absolute_path.has_parent_path()) {
                            if (absolute_path.has_parent_path()) {
                                this->file_path_str = absolute_path.parent_path().parent_path().generic_u8string();
                            }
                        }
                    }

                    // Update child paths
                    this->child_paths.clear();

                    std::vector<ChildData_t> paths;
                    std::vector<ChildData_t> files;
                    try {
                        stdfs::path tmp_file_path = static_cast<stdfs::path>(this->file_path_str);
                        for (const auto& entry : stdfs::directory_iterator(tmp_file_path)) {
                            if (stdfs::status_known(stdfs::status(entry.path()))) {
                                bool is_directory = stdfs::is_directory(entry.path());
                                if (is_directory) {
                                    paths.emplace_back(ChildData_t(entry.path(), is_directory));
                                } else {
                                    files.emplace_back(ChildData_t(entry.path(), is_directory));
                                }
                            }
                        }
                    } catch (...) {
                    }

                    // Sort path case insensitive alphabetically ascending
                    std::sort(paths.begin(), paths.end(), [](ChildData_t const& a, ChildData_t const& b) {
                        std::string a_str = a.first.filename().generic_u8string();
                        for (auto& c : a_str) c = std::toupper(c);
                        std::string b_str = b.first.filename().generic_u8string();
                        for (auto& c : b_str) c = std::toupper(c);
                        return (a_str < b_str);
                    });
                    std::sort(files.begin(), files.end(), [](ChildData_t const& a, ChildData_t const& b) {
                        std::string a_str = a.first.filename().generic_u8string();
                        for (auto& c : a_str) c = std::toupper(c);
                        std::string b_str = b.first.filename().generic_u8string();
                        for (auto& c : b_str) c = std::toupper(c);
                        return (a_str < b_str);
                    });

                    for (auto& path : paths) {
                        this->child_paths.emplace_back(path);
                    }
                    for (auto& file : files) {
                        this->child_paths.emplace_back(file);
                    }

                    this->path_changed = false;
                }

                // Files and directories ----------------
                for (const auto& path_pair : this->child_paths) {

                    auto select_label = path_pair.first.filename().generic_u8string();
                    bool showSearchedParameter = true;
                    if (!currentSearchString.empty()) {
                        showSearchedParameter =
                            StringSearchWidget::FindCaseInsensitiveSubstring(select_label, currentSearchString);
                    }
                    if (showSearchedParameter) {
                        // Different color for directories
                        if (path_pair.second) {
                            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
                        }

                        if (ImGui::Selectable(
                                select_label.c_str(), (select_label == this->file_name_str), select_flags)) {
                            last_file_path_str = this->file_path_str;
                            this->splitPath(path_pair.first, this->file_path_str, this->file_name_str);
                            this->validateFile(this->file_name_str, flag);
                            if (last_file_path_str != this->file_path_str) {
                                this->path_changed = true;
                            }
                        }

                        if (path_pair.second) {
                            ImGui::PopStyleColor();
                        }
                        if (ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
                            apply = true;
                        }
                    }
                }
            }
            ImGui::EndChild();

            // Widget group -------------------------------------------------------
            ImGui::BeginGroup();

            // File name ------------------------
            if (flag == FileBrowserFlag::LOAD) {
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            }
            auto last_file_name_str = this->file_name_str;
            /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
            GUIUtils::Utf8Encode(this->file_name_str);
            if (ImGui::InputText("File Name", &this->file_name_str,
                    ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll)) {
                apply = true;
            }
            GUIUtils::Utf8Decode(this->file_name_str);
            if (flag == FileBrowserFlag::LOAD) {
                ImGui::PopItemFlag();
            }
            if (last_file_name_str != this->file_name_str) {
                this->validateFile(this->file_name_str, flag);
            }

            // Buttons ------------------------------
            std::string button_label;
            if (flag == FileBrowserFlag::SAVE) {
                button_label = "Save";
            } else if (flag == FileBrowserFlag::LOAD) {
                button_label = "Load";
            } else if (flag == FileBrowserFlag::SELECT) {
                button_label = "Select";
            }
            if (ImGui::Button(button_label.c_str())) {
                apply = true;
            }

            ImGui::SameLine();

            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
            }

            // Error and warn massages
            if (!this->file_warning.empty()) {
                ImGui::TextColored(GUI_COLOR_TEXT_WARN, this->file_warning.c_str());
            }
            if (!this->file_error.empty()) {
                ImGui::TextColored(GUI_COLOR_TEXT_ERROR, this->file_error.c_str());
            }

            // Return complete file path --------------------------------------
            if (apply && this->valid_directory && this->valid_file) {
                // Appending required extension
                if (!this->valid_ending) {
                    this->file_name_str.append(".lua");
                }
                stdfs::path tmp_file_path =
                    static_cast<stdfs::path>(this->file_path_str) / static_cast<stdfs::path>(this->file_name_str);
                inout_filename = tmp_file_path.generic_u8string();
                ImGui::CloseCurrentPopup();
                retval = true;
            }

            ImGui::EndGroup();
            // --------------------------------------------------------------------

            ImGui::EndPopup();
        }
#endif // GUI_USE_FILESYSTEM

        ImGui::PopID();

    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return retval;
}


bool megamol::gui::FileBrowserWidget::Button(std::string& inout_filename) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    float edge_length = ImGui::GetFrameHeight();
    float half_edge_length = edge_length / 2.0f;
    ImVec2 widget_start_pos = ImGui::GetCursorScreenPos();

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBg]));
    ImGui::BeginChild("filebrowser_button_background", ImVec2(edge_length, edge_length), false,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    float width = half_edge_length * 0.7f;
    float height = width * 0.7f;

    ImVec2 center = widget_start_pos + ImVec2(half_edge_length, half_edge_length);
    ImVec2 upper_left = center - ImVec2(width, height);
    ImVec2 lower_right = center + ImVec2(width, height);
    ImVec4 color_front = style.Colors[ImGuiCol_ButtonActive];
    color_front.w = 1.0f;
    draw_list->AddRectFilled(upper_left, lower_right, ImGui::ColorConvertFloat4ToU32(color_front), 1.0f);

    center += ImVec2(half_edge_length * 0.25f, half_edge_length * 0.25f);
    upper_left = center - ImVec2(width, height);
    lower_right = center + ImVec2(width, height);
    color_front = style.Colors[ImGuiCol_ButtonHovered];
    color_front.w = 1.0f;
    draw_list->AddRectFilled(upper_left, lower_right, ImGui::ColorConvertFloat4ToU32(color_front), 1.0f);

    ImVec2 rect = ImVec2(edge_length, edge_length);
    bool popup_select_file = ImGui::InvisibleButton("special_button", rect);

    ImGui::EndChild();
    ImGui::PopStyleColor();

    bool retval = this->PopUp(FileBrowserFlag::SELECT, "Select File", popup_select_file, inout_filename);

    return retval;
}


#ifdef GUI_USE_FILESYSTEM

bool megamol::gui::FileBrowserWidget::splitPath(
    const stdfs::path& in_file_path, std::string& out_path, std::string& out_file) {

    // Splitting path into path string and file string
    try {
        if (stdfs::is_regular_file(in_file_path)) {
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
    } catch (stdfs::filesystem_error e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


void megamol::gui::FileBrowserWidget::validateDirectory(const std::string& path_str) {

    // Validating directory
    try {
        stdfs::path tmp_path = static_cast<stdfs::path>(path_str);
        this->valid_directory = (stdfs::status_known(stdfs::status(tmp_path)) && stdfs::is_directory(tmp_path));
    } catch (stdfs::filesystem_error e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


void megamol::gui::FileBrowserWidget::validateFile(
    const std::string& file_str, megamol::gui::FileBrowserWidget::FileBrowserFlag flag) {

    // Validating file
    try {
        const std::string ext(".lua");
        this->file_error.clear();
        this->file_warning.clear();
        this->additional_lines = 0;
        this->valid_file = true;
        this->valid_ending = true;

        stdfs::path tmp_file_path = static_cast<stdfs::path>(this->file_path_str) / static_cast<stdfs::path>(file_str);

        if (flag == FileBrowserFlag::SAVE) {
            // Warn when no file name is given
            if (file_str.empty()) {
                this->file_warning += "Enter file name.\n";
                this->additional_lines++;
                this->valid_file = false;
            } else {
                // Warn when file has not required extension
                if (!FileUtils::FileExtension<std::string>(file_str, ext)) {
                    this->file_warning += "Appending required file extension '" + ext + "'\n";
                    this->additional_lines++;
                    this->valid_ending = false;
                }
                std::string actual_filename = file_str;
                if (!this->valid_ending) {
                    actual_filename.append(ext);
                }
                tmp_file_path =
                    static_cast<stdfs::path>(this->file_path_str) / static_cast<stdfs::path>(actual_filename);

                // Warn when file already exists
                if (stdfs::exists(tmp_file_path) && stdfs::is_regular_file(tmp_file_path)) {
                    this->file_warning += "Overwriting existing file.\n";
                    this->additional_lines++;
                }
            }
        } else if (flag == FileBrowserFlag::LOAD) {
            // Error when file has not required extension
            if (!FileUtils::FileExtension<std::string>(file_str, ext)) {
                this->file_error += "File with extension '" + ext + "' required.\n";
                this->additional_lines++;
                this->valid_ending = false;
                this->valid_file = false;
            }
        } else if (flag == FileBrowserFlag::SELECT) {
            // nothing to check ...
        }

        // Error when file is directory
        if (stdfs::is_directory(tmp_file_path)) {
            this->file_error += "File is directory.\n";
            this->additional_lines++;
            this->valid_file = false;
        }

    } catch (stdfs::filesystem_error e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}

#endif // GUI_USE_FILESYSTEM
