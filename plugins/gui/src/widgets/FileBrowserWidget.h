/*
 * FileBrowserWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_FILEBROWSERPOPUP_INCLUDED
#define MEGAMOL_GUI_FILEBROWSERPOPUP_INCLUDED
#pragma once


#include "HoverToolTip.h"
#include "StringSearchWidget.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/FileUtils.h"


namespace megamol {
namespace gui {


    /** ************************************************************************
     * File browser widget
     */
    class FileBrowserWidget {
    public:
        FileBrowserWidget();
        ~FileBrowserWidget() = default;

        /**
         * Show file browser pop-up.
         *
         * @param label                 The pop-up label.
         * @param extension             The required file extension. Leave emtpy to allow all file extensions.
         * @param inout_open_popup      Indicates whether to open the pop-up or not.
         * @param inout_filename        The user selected file name. Provide file name to use as initial value.
         * @param force_absolute        Indicates whether to return the absolute file path or to enable relative path
         * selection.
         * @param project_path          The path to the current project file that is used for determine relative path.
         * @param inout_save_gui_state  The flag indicating whether to save the gui state or not.
         *
         * @return True on success, false otherwise.
         */
        bool PopUp_Save(const std::string& label, const std::vector<std::string>& extensions, bool& inout_open_popup,
            std::string& inout_filename, vislib::math::Ternary& inout_save_gui_state) {
            return this->popup(
                DIALOGMODE_SAVE, label, extensions, inout_open_popup, inout_filename, false, "", inout_save_gui_state);
        }
        bool PopUp_Load(const std::string& label, const std::vector<std::string>& extensions, bool& inout_open_popup,
            std::string& inout_filename) {
            auto tmp = vislib::math::Ternary(vislib::math::Ternary::TRI_UNKNOWN);
            return this->popup(DIALOGMODE_LOAD, label, extensions, inout_open_popup, inout_filename, false, "", tmp);
        }
        bool PopUp_Select(const std::string& label, const std::vector<std::string>& extensions, bool& inout_open_popup,
            std::string& inout_filename, bool force_absolute, const std::string& project_path) {
            auto tmp = vislib::math::Ternary(vislib::math::Ternary::TRI_UNKNOWN);
            return this->popup(DIALOGMODE_SELECT, label, extensions, inout_open_popup, inout_filename, force_absolute,
                project_path, tmp);
        }

        /**
         * ImGui file browser button opening this file browser pop-up.
         *
         * @param extensions       The filtered file extensions. Leave emtpy to allow all file extensions.
         * @param inout_filename   The user selected file name.
         * @param force_absolute   The flag that indicates to whether to return the absolute file path or to enable
         * relative path selection.
         * @param project_path     The path to the current project file that is used for determine relative path.
         *
         * @return True on success, false otherwise.
         */
        bool Button_Select(const std::vector<std::string>& extensions, std::string& inout_filename, bool force_absolute,
            const std::string& project_path = "");

    private:
        typedef std::pair<stdfs::path, bool> ChildData_t;

        enum DialogMode { DIALOGMODE_SAVE, DIALOGMODE_LOAD, DIALOGMODE_SELECT };
        enum PathMode { PATHMODE_RELATIVE_PROJECT, PATHMODE_RELATIVE_WORKING, PATHMODE_ABSOLUTE };

        // VARIABLES --------------------------------------------------------------

        StringSearchWidget search_widget;
        std::string file_name_str;
        std::string file_path_str;
        bool path_changed;
        bool valid_directory;
        bool valid_file;
        std::string valid_ending;
        std::string file_error;
        std::string file_warning;
        // Keeps child path and flag whether child is director or not
        std::vector<ChildData_t> child_paths;
        vislib::math::Ternary save_gui_state;
        std::map<std::string, std::string> label_uid_map;
        PathMode return_path;

        HoverToolTip tooltip;

        // FUNCTIONS --------------------------------------------------------------

        bool popup(DialogMode mode, const std::string& label, const std::vector<std::string>& extensions,
            bool& inout_open_popup, std::string& inout_filename, bool force_absolute_path,
            const std::string& project_path, vislib::math::Ternary& inout_save_gui_state);

        bool validate_split_path(const std::string& in_path_file, std::string& out_path, std::string& out_file) const;

        void validate_directory(const std::string& path_str);

        void validate_file(const std::string& file_str, const std::vector<std::string>& extensions, DialogMode mode);

        std::string get_absolute_path(const std::string& in_path_str) const;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_FileBrowserPopUp_INCLUDED
