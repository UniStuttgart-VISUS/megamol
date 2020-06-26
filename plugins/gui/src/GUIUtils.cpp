/*
 * PopUp.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GUIUtils.h"


using namespace megamol;
using namespace megamol::gui;


ImGuiID megamol::gui::gui_generated_uid = 0;


GUIUtils::GUIUtils(void)
    : tooltip_time(0.0f)
    , tooltip_id(GUI_INVALID_ID)
    , search_focus(false)
    , search_string()
    , rename_string()
    , splitter_last_width(0.0f) {}


bool GUIUtils::HoverToolTip(const std::string& text, ImGuiID id, float time_start, float time_end) {
    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiIO& io = ImGui::GetIO();

    bool retval = false;

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
            retval = true;
        }
    } else {
        if ((time_start > 0.0f) && (this->tooltip_id == id)) {
            this->tooltip_time = 0.0f;
        }
    }

    return retval;
}


void GUIUtils::ResetHoverToolTip(void) {

    this->tooltip_time = 0.0f;
    this->tooltip_id = GUI_INVALID_ID;
}


bool GUIUtils::HelpMarkerToolTip(const std::string& text, std::string label) {
    assert(ImGui::GetCurrentContext() != nullptr);

    if (!text.empty()) {
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled(label.c_str());
        return this->HoverToolTip(text);
    }
    return false;
}


bool megamol::gui::GUIUtils::MinimalPopUp(const std::string& caption, bool open_popup, const std::string& info_text,
    const std::string& confirm_btn_text, bool& confirmed, const std::string& abort_btn_text, bool& aborted) {

    bool retval = false;

    ImGuiStyle& style = ImGui::GetStyle();

    ImGui::PushID(caption.c_str());

    if (open_popup) {
        ImGui::OpenPopup(caption.c_str());
        float max_width = std::max(ImGui::CalcTextSize(caption.c_str()).x, ImGui::CalcTextSize(info_text.c_str()).x);
        max_width += (style.ItemSpacing.x * 2.0f);
        ImGui::SetNextWindowSize(ImVec2(max_width, 0.0f));
    }
    if (ImGui::BeginPopupModal(caption.c_str(), nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {
        retval = true;

        if (!info_text.empty()) {
            ImGui::TextUnformatted(info_text.c_str());
        }

        if (!confirm_btn_text.empty()) {
            if (ImGui::Button(confirm_btn_text.c_str())) {
                confirmed = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
        }
        if (!abort_btn_text.empty()) {
            if (ImGui::Button(abort_btn_text.c_str())) {
                aborted = true;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }

    ImGui::PopID();

    return retval;
}


bool megamol::gui::GUIUtils::RenamePopUp(const std::string& caption, bool open_popup, std::string& rename) {

    bool retval = false;

    ImGui::PushID(caption.c_str());

    if (open_popup) {
        this->rename_string = rename;
        ImGui::OpenPopup(caption.c_str());
    }
    if (ImGui::BeginPopupModal(caption.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {

        std::string text_label = "New Name";
        auto flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll;
        if (ImGui::InputText(text_label.c_str(), &this->rename_string, flags)) {
            rename = this->rename_string;
            retval = true;
            ImGui::CloseCurrentPopup();
        }
        // Set focus on input text once (applied next frame)
        if (open_popup) {
            ImGuiID id = ImGui::GetID(text_label.c_str());
            ImGui::ActivateItem(id);
        }

        if (ImGui::Button("OK")) {
            rename = this->rename_string;
            retval = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    ImGui::PopID();

    return retval;
}


bool GUIUtils::Utf8Decode(std::string& str) {
    vislib::StringA dec_tmp;
    if (vislib::UTF8Encoder::Decode(dec_tmp, vislib::StringA(str.c_str()))) {
        str = std::string(dec_tmp.PeekBuffer());
        return true;
    }
    return false;
}


bool GUIUtils::Utf8Encode(std::string& str) {
    vislib::StringA dec_tmp;
    if (vislib::UTF8Encoder::Encode(dec_tmp, vislib::StringA(str.c_str()))) {
        str = std::string(dec_tmp.PeekBuffer());
        return true;
    }
    return false;
}


bool megamol::gui::GUIUtils::StringSearch(const std::string& id, const std::string& help) {

    bool retval = false;

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
        this->search_string = "";
        this->search_focus = false;
    }

    std::string complete_label = "Search (?)";
    auto width = ImGui::GetContentRegionAvail().x - ImGui::GetCursorPosX() + 4.0f * style.ItemInnerSpacing.x -
                 ImGui::CalcTextSize(complete_label.c_str()).x;
    const float min_width = 50.0f;
    width = (width < min_width) ? (min_width) : width;
    ImGui::PushItemWidth(width);
    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
    this->Utf8Encode(this->search_string);
    ImGui::InputText("Search", &this->search_string, ImGuiInputTextFlags_AutoSelectAll);
    this->Utf8Decode(this->search_string);
    if (ImGui::IsItemActive()) {
        retval = true;
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();

    this->HelpMarkerToolTip(help);

    ImGui::PopID();
    ImGui::EndGroup();

    return retval;
}


bool megamol::gui::GUIUtils::VerticalSplitter(FixedSplitterSide fixed_side, float& size_left, float& size_right) {
    assert(ImGui::GetCurrentContext() != nullptr);

    const float thickness = 12.0f;

    bool split_vertically = true;
    float min_size = 1.0f; // >=1.0!
    float splitter_long_axis_size = ImGui::GetContentRegionAvail().y;

    float width_avail = ImGui::GetWindowSize().x - (2.0f * thickness);

    size_left = ((fixed_side == GUIUtils::FixedSplitterSide::LEFT) ? size_left : (width_avail - size_right));
    size_right = ((fixed_side == GUIUtils::FixedSplitterSide::LEFT) ? (width_avail - size_left) : size_right);

    size_left = std::max(size_left, min_size);
    size_right = std::max(size_right, min_size);

    ImGuiWindow* window = ImGui::GetCurrentContext()->CurrentWindow;
    ImGuiID id = window->GetID("##Splitter");
    ImRect bb;
    if (fixed_side == GUIUtils::FixedSplitterSide::LEFT) {
        bb.Min =
            window->DC.CursorPos + (split_vertically ? ImVec2(size_left + 1.0f, 0.0f) : ImVec2(0.0f, size_left + 1.0f));
    } else if (fixed_side == GUIUtils::FixedSplitterSide::RIGHT) {
        bb.Min = window->DC.CursorPos + (split_vertically ? ImVec2((width_avail - size_right) + 1.0f, 0.0f)
                                                          : ImVec2(0.0f, (width_avail - size_right) + 1.0f));
    }
    bb.Max = bb.Min + ImGui::CalcItemSize(split_vertically ? ImVec2(thickness - 6.0f, splitter_long_axis_size)
                                                           : ImVec2(splitter_long_axis_size, thickness - 6.0f),
                          0.0f, 0.0f);

    bool retval = ImGui::SplitterBehavior(
        bb, id, split_vertically ? ImGuiAxis_X : ImGuiAxis_Y, &size_left, &size_right, min_size, min_size, 0.0f, 0.0f);

    /// XXX Left mouse button (= 0) is not recognized poperly!? ...
    if (ImGui::IsMouseDoubleClicked(1) && ImGui::IsItemHovered()) {
        float consider_width = ((fixed_side == GUIUtils::FixedSplitterSide::LEFT) ? size_left : size_right);
        if (consider_width <= min_size) {
            size_left = ((fixed_side == GUIUtils::FixedSplitterSide::LEFT) ? (this->splitter_last_width)
                                                                           : (width_avail - this->splitter_last_width));
            size_right = ((fixed_side == GUIUtils::FixedSplitterSide::LEFT) ? (width_avail - this->splitter_last_width)
                                                                            : (this->splitter_last_width));
        } else {
            size_left = ((fixed_side == GUIUtils::FixedSplitterSide::LEFT) ? (min_size) : (width_avail - min_size));
            size_right = ((fixed_side == GUIUtils::FixedSplitterSide::LEFT) ? (width_avail - min_size) : (min_size));
            this->splitter_last_width = consider_width;
        }
    }

    return retval;
}


bool megamol::gui::GUIUtils::LoadTexture(const std::string& filename, GLuint& inout_id) {

    if (filename.empty()) return false;
    bool retval = false;

    static vislib::graphics::BitmapImage img;
    static sg::graphics::PngBitmapCodec pbc;
    pbc.Image() = &img;
    void* buf = nullptr;
    size_t size = 0;

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    size = megamol::gui::GUIUtils::LoadRawFile(filename, &buf);
    if (size > 0) {
        if (pbc.Load(buf, size)) {
            img.Convert(vislib::graphics::BitmapImage::TemplateFloatRGBA);
            retval =
                megamol::gui::GUIUtils::CreateTexture(inout_id, img.Width(), img.Height(), img.PeekDataAs<FLOAT>());
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to read texture: %s [%s, %s, line %d]\n", filename.c_str(), __FILE__, __FUNCTION__, __LINE__);
            retval = false;
        }
    } else {
        retval = false;
    }

    ARY_SAFE_DELETE(buf);
    return retval;
}


size_t megamol::gui::GUIUtils::LoadRawFile(std::string name, void** outData) {

    *outData = nullptr;

    vislib::StringW filename = static_cast<vislib::StringW>(name.c_str());
    if (filename.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Unable to load file: No file name given. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }

    if (!vislib::sys::File::Exists(filename)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to load file \"%s\": Not existing. [%s, %s, line %d]\n",
            name.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }

    size_t size = static_cast<size_t>(vislib::sys::File::GetSize(filename));
    if (size < 1) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to load file \"%s\": File is empty. [%s, %s, line %d]\n",
            name.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }

    vislib::sys::FastFile f;
    if (!f.Open(filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to load file \"%s\": Unable to open file. [%s, %s, line %d]\n",
            name.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }

    *outData = new BYTE[size];
    size_t num = static_cast<size_t>(f.Read(*outData, size));
    if (num != size) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Unable to load file \"%s\": Unable to read whole file. [%s, %s, line %d]\n", name.c_str(), __FILE__,
            __FUNCTION__, __LINE__);
        ARY_SAFE_DELETE(*outData);
        return 0;
    }

    return num;
}


bool megamol::gui::GUIUtils::CreateTexture(GLuint& inout_id, GLsizei width, GLsizei height, const float* data) {
    if (data == nullptr) return false;

    // Delete old texture.
    if (inout_id != 0) {
        glDeleteTextures(1, &inout_id);
    }
    inout_id = 0;

    // Upload texture.
    glGenTextures(1, &inout_id);
    glBindTexture(GL_TEXTURE_2D, inout_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, data);

    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}


void megamol::gui::GUIUtils::ReadOnlyWigetStyle(bool set) {

    if (set) {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    } else {
        ImGui::PopItemFlag();
        ImGui::PopStyleVar();
    }
}