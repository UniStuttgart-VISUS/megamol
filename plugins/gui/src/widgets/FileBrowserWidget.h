/*
 * FileBrowserWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_FILEBROWSERPOPUP_INCLUDED
#define MEGAMOL_GUI_FILEBROWSERPOPUP_INCLUDED


#include "GUIUtils.h"
#include "HoverToolTip.h"
#include "StringSearchWidget.h"

#include "mmcore/utility/FileUtils.h"


namespace megamol {
namespace gui {


    /**
     * String search widget.
     */
    class FileBrowserWidget {
    public:
        FileBrowserWidget();

        ~FileBrowserWidget() = default;

        enum FileBrowserFlag { SAVE, LOAD, SELECT };

        /**
         * Draw file browser pop-up.
         *
         * @param inout_filename      The file name of the file.
         * @param flag                Flag inidicating intention of file browser dialog.
         * @param label               File browser label.
         * @param open_popup          Flag once(!) indicates opening of pop-up.
         * @param extension           The file name extension.
         *
         * @return True on success, false otherwise.
         */
        bool PopUp(std::string& inout_filename, FileBrowserFlag flag, const std::string& label, bool open_popup,
            const std::string& extension, vislib::math::Ternary& inout_save_gui_state);

        bool PopUp(std::string& inout_filename, FileBrowserFlag flag, const std::string& label, bool open_popup,
            const std::string& extension) {
            vislib::math::Ternary tmp_save_gui_state(vislib::math::Ternary::TRI_UNKNOWN);
            return this->PopUp(inout_filename, flag, label, open_popup, extension, tmp_save_gui_state);
        }

        /**
         * ImGui file browser button opening this file browser pop-up.
         *
         * @param inout_filename    The file name of the file.
         *
         * @return True on success, false otherwise.
         */
        bool Button(std::string& inout_filename, FileBrowserFlag flag, const std::string& extension);

    private:
        typedef std::pair<stdfs::path, bool> ChildData_t;

        // VARIABLES --------------------------------------------------------------

        StringSearchWidget search_widget;
        std::string file_name_str;
        std::string file_path_str;
        bool path_changed;
        bool valid_directory;
        bool valid_file;
        bool valid_ending;
        std::string file_error;
        std::string file_warning;
        // Keeps child path and flag whether child is director or not
        std::vector<ChildData_t> child_paths;
        size_t additional_lines;
        vislib::math::Ternary save_gui_state;

        HoverToolTip tooltip;

        // FUNCTIONS --------------------------------------------------------------

        bool validate_split_path(const std::string& in_path_file, std::string& out_path, std::string& out_file);
        void validate_directory(const std::string& path_str);
        void validate_file(const std::string& file_str, const std::string& extension, FileBrowserFlag flag);
        std::string get_absolute_path(const std::string& in_path_str) const;

        void string_to_lower_case(std::string& str);
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_FileBrowserPopUp_INCLUDED
