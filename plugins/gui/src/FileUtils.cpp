/*
 * FileUtils.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "FileUtils.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::FileUtils::FileUtils(void)
#ifdef GUI_USE_FILESYSTEM
    : utils()
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
    vislib::sys::Log::DefaultLog.WriteWarn(
        "Filesystem functionality is not available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
#endif // GUI_USE_FILESYSTEM
}


bool megamol::gui::FileUtils::SaveProjectFile(
    const std::string& project_filename, megamol::core::CoreInstance* core_instance) {
#ifdef GUI_USE_FILESYSTEM
    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to CoreInstance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    std::string serInstances, serModules, serCalls, serParams;
    core_instance->SerializeGraph(serInstances, serModules, serCalls, serParams);
    auto confstr = serInstances + "\n" + serModules + "\n" + serCalls + "\n" + serParams + "\n";

    try {
        std::ofstream file;
        file.open(project_filename);
        if (file.good()) {
            file << confstr.c_str();
            file.close();
        } else {
            vislib::sys::Log::DefaultLog.WriteError(
                "Unable to create project file. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            file.close();
            return false;
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
#else
    return false;
#endif // GUI_USE_FILESYSTEM
}


bool megamol::gui::FileUtils::WriteFile(const std::string& filename, const std::string& in_content) {
#ifdef GUI_USE_FILESYSTEM
    try {
        std::ofstream file;
        file.open(filename, std::ios_base::out);
        if (file.is_open() && file.good()) {
            file << in_content.c_str();
            file.close();
        } else {
            vislib::sys::Log::DefaultLog.WriteError(
                "Unable to create file. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            file.close();
            return false;
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
#else
    return false;
#endif // GUI_USE_FILESYSTEM
}


bool megamol::gui::FileUtils::ReadFile(const std::string& filename, std::string& out_content) {
#ifdef GUI_USE_FILESYSTEM
    try {
        std::ifstream file;
        file.open(filename, std::ios_base::in);
        if (file.is_open() && file.good()) {
            out_content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            file.close();
        } else {
            vislib::sys::Log::DefaultLog.WriteError(
                "Unable to open file. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            file.close();
            return false;
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
#else
    return false;
#endif // GUI_USE_FILESYSTEM
}


bool megamol::gui::FileUtils::FileBrowserButton(std::string& out_filename) {

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

    return this->FileBrowserPopUp(FileBrowserFlag::SELECT, "Select File", popup_select_file, out_filename);
}


bool megamol::gui::FileUtils::FileBrowserPopUp(megamol::gui::FileUtils::FileBrowserFlag flag, const std::string& label,
    bool open_popup, std::string& out_filename) {

    bool retval = false;

    try {
        std::string popup_name = label;
        ImGui::PushID(label.c_str());

        if (open_popup) {
#ifdef GUI_USE_FILESYSTEM
            // Check given file name path
            fsns::path tmp_file_path = static_cast<fsns::path>(out_filename);
            if (tmp_file_path.empty() || !fsns::exists(tmp_file_path)) {
                tmp_file_path = fsns::current_path();
            }
            this->splitPath(tmp_file_path, this->file_path_str, this->file_name_str);
            this->validateDirectory(this->file_path_str);
            this->validateFile(this->file_name_str, flag);
            this->path_changed = true;

            ImGui::OpenPopup(popup_name.c_str());
            // Set initial window size of pop up
            ImGui::SetNextWindowSize(ImVec2(600.0f, 300.0f));
#else
            vislib::sys::Log::DefaultLog.WriteWarn(
                "Filesystem functionality is not available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
#endif // GUI_USE_FILESYSTEM
        }

#ifdef GUI_USE_FILESYSTEM
        bool open = true;
        if (ImGui::BeginPopupModal(popup_name.c_str(), &open, ImGuiWindowFlags_None)) {

            bool apply = false;
            const auto error_color = ImVec4(0.9f, 0.0f, 0.0f, 1.0f);
            const auto warning_color = ImVec4(0.75f, 0.75f, 0.f, 1.0f);

            // Path ---------------------------------------------------
            auto last_file_path_str = this->file_path_str;
            if (ImGui::ArrowButton("###arrow_up_directory", ImGuiDir_Up)) {
                fsns::path tmp_file_path = static_cast<fsns::path>(this->file_path_str);
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
                ImGui::TextColored(error_color, "Invalid Directory");
            }

            // Search -------------------------------
            std::string help_test = "Case insensitive substring search in\nlisted file and directory names.";
            this->utils.StringSearch("guiwindow_parameter_earch", help_test);
            auto currentSearchString = this->utils.GetSearchString();

            // File browser selectables ---------------------------------------
            auto select_flags = ImGuiSelectableFlags_DontClosePopups;
            float child_select_height =
                (ImGui::GetContentRegionAvail().y - (ImGui::GetFrameHeightWithSpacing() * this->additional_lines) -
                    ImGui::GetFrameHeightWithSpacing() * 2.0f);
            ImGui::BeginChild(
                "files_list_child_window", ImVec2(0.0f, child_select_height), true, ImGuiWindowFlags_None);

            if (this->valid_directory) {
                // Parent directory selectable
                std::string tag_parent = "..";
                if (ImGui::Selectable(tag_parent.c_str(), false, select_flags)) {
                    fsns::path tmp_file_path = static_cast<fsns::path>(this->file_path_str);
                    if (tmp_file_path.has_parent_path()) {
                        // Assuming that parent is still valid directory
                        this->file_path_str = tmp_file_path.parent_path().generic_u8string();
                        this->path_changed = true;
                    }
                }

                // Only update child paths when path changed.
                if (this->path_changed) {
                    this->child_paths.clear();
                    try {
                        fsns::path tmp_file_path = static_cast<fsns::path>(this->file_path_str);
                        for (const auto& entry : fsns::directory_iterator(tmp_file_path)) {
                            if (fsns::status_known(fsns::status(entry.path()))) {
                                this->child_paths.emplace_back(
                                    ChildDataType(entry.path(), fsns::is_directory(entry.path())));
                            }
                        }
                    } catch (...) {
                    }

                    // Sort path case insensitive alphabetically ascending
                    std::sort(this->child_paths.begin(), this->child_paths.end(),
                        [](ChildDataType const& a, ChildDataType const& b) {
                            std::string a_str = a.first.filename().generic_u8string();
                            for (auto& c : a_str) c = std::toupper(c);
                            std::string b_str = b.first.filename().generic_u8string();
                            for (auto& c : b_str) c = std::toupper(c);
                            return (a_str < b_str);
                        });

                    this->path_changed = false;
                }

                // Files and directories ----------------
                for (const auto& path_pair : this->child_paths) {

                    auto label = path_pair.first.filename().generic_u8string();
                    bool showSearchedParameter = true;
                    if (!currentSearchString.empty()) {
                        showSearchedParameter = this->utils.FindCaseInsensitiveSubstring(label, currentSearchString);
                    }
                    if (showSearchedParameter) {
                        // Different color for directories
                        if (path_pair.second) {
                            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
                        }

                        if (ImGui::Selectable(label.c_str(), (label == this->file_name_str), select_flags)) {
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
            if (flag == FileUtils::FileBrowserFlag::LOAD) {
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
            if (flag == FileUtils::FileBrowserFlag::LOAD) {
                ImGui::PopItemFlag();
            }
            if (last_file_name_str != this->file_name_str) {
                this->validateFile(this->file_name_str, flag);
            }
            if (!this->file_warning.empty()) {
                ImGui::TextColored(warning_color, this->file_warning.c_str());
            }
            if (!this->file_error.empty()) {
                ImGui::TextColored(error_color, this->file_error.c_str());
            }

            // Buttons ------------------------------
            std::string label;
            if (flag == FileUtils::FileBrowserFlag::SAVE) {
                label = "Save";
            } else if (flag == FileUtils::FileBrowserFlag::LOAD) {
                label = "Load";
            } else if (flag == FileUtils::FileBrowserFlag::SELECT) {
                label = "Select";
            }
            // const auto err_btn_color = ImVec4(0.6f, 0.0f, 0.0f, 1.0f);
            // const auto er_btn_hov_color = ImVec4(0.9f, 0.0f, 0.0f, 1.0f);
            // ImGui::PushStyleColor(ImGuiCol_Button,
            //     !(this->valid_directory && this->valid_file) ? err_btn_color : style.Colors[ImGuiCol_Button]);
            // ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
            //     !(this->valid_directory && this->valid_file) ? er_btn_hov_color :
            //     style.Colors[ImGuiCol_ButtonHovered]);
            // ImGui::PushStyleColor(ImGuiCol_ButtonActive, style.Colors[ImGuiCol_ButtonActive]);
            if (ImGui::Button(label.c_str())) {
                apply = true;
            }
            // ImGui::PopStyleColor(3);

            ImGui::SameLine();

            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
            }

            // Return complete file path --------------------------------------
            if (apply && this->valid_directory && this->valid_file) {
                // Appending required extension
                if (!this->valid_ending) {
                    this->file_name_str.append(".lua");
                }
                fsns::path tmp_file_path =
                    static_cast<fsns::path>(this->file_path_str) / static_cast<fsns::path>(this->file_name_str);
                out_filename = tmp_file_path.generic_u8string();
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
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return retval;
}


#ifdef GUI_USE_FILESYSTEM

bool megamol::gui::FileUtils::splitPath(const fsns::path& in_file_path, std::string& out_path, std::string& out_file) {

    // Splitting path into path string and file string
    try {
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
    } catch (fsns::filesystem_error e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


void megamol::gui::FileUtils::validateDirectory(const std::string& path_str) {

    // Validating directory
    try {
        fsns::path tmp_path = static_cast<fsns::path>(path_str);
        this->valid_directory = (fsns::status_known(fsns::status(tmp_path)) && fsns::is_directory(tmp_path));
    } catch (fsns::filesystem_error e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


void megamol::gui::FileUtils::validateFile(const std::string& file_str, megamol::gui::FileUtils::FileBrowserFlag flag) {

    // Validating file
    try {
        const std::string ext = ".lua";
        this->file_error.clear();
        this->file_warning.clear();
        this->additional_lines = 0;
        this->valid_file = true;
        this->valid_ending = true;

        fsns::path tmp_file_path = static_cast<fsns::path>(this->file_path_str) / static_cast<fsns::path>(file_str);

        if (flag == FileUtils::FileBrowserFlag::SAVE) {
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
                tmp_file_path = static_cast<fsns::path>(this->file_path_str) / static_cast<fsns::path>(actual_filename);

                // Warn when file already exists
                if (fsns::exists(tmp_file_path) && fsns::is_regular_file(tmp_file_path)) {
                    this->file_warning += "Overwriting existing file.\n";
                    this->additional_lines++;
                }
            }
        } else if (flag == FileUtils::FileBrowserFlag::LOAD) {
            // Error when file has not required extension
            if (!FileUtils::FileExtension<std::string>(file_str, ext)) {
                this->file_error += "File with extension '" + ext + "' required.\n";
                this->additional_lines++;
                this->valid_ending = false;
                this->valid_file = false;
            }
        } else if (flag == FileUtils::FileBrowserFlag::SELECT) {
            // nothing to check ...
        }

        // Error when file is directory
        if (fsns::is_directory(tmp_file_path)) {
            this->file_error += "File is directory.\n";
            this->additional_lines++;
            this->valid_file = false;
        }

    } catch (fsns::filesystem_error e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}

#endif // GUI_USE_FILESYSTEM
