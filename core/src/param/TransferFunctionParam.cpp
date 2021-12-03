/*
 * TransferFunctionParam.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/param/TransferFunctionParam.h"
#include "stdafx.h"


using namespace megamol::core::param;


TransferFunctionParam::TransferFunctionParam(const std::string& initVal) : AbstractParam(), val(), hash(0) {
    this->InitPresentation(AbstractParamPresentation::ParamType::TRANSFERFUNCTION);
    this->SetValue(initVal);
}


TransferFunctionParam::TransferFunctionParam(const char* initVal) : AbstractParam(), val(), hash(0) {
    this->InitPresentation(AbstractParamPresentation::ParamType::TRANSFERFUNCTION);
    this->SetValue(std::string(initVal));
}


TransferFunctionParam::TransferFunctionParam(const vislib::StringA& initVal) : AbstractParam(), val(), hash(0) {
    this->InitPresentation(AbstractParamPresentation::ParamType::TRANSFERFUNCTION);
    this->SetValue(std::string(initVal.PeekBuffer()));
}


std::string TransferFunctionParam::Definition() const {
    return "MMTFFC";
}


bool TransferFunctionParam::ParseValue(std::string const& v) {
    try {
        this->SetValue(v);
        return true;
    } catch (...) {}
    return false;
}


void TransferFunctionParam::SetValue(const std::string& v, bool setDirty) {

    if (v != this->val) {
        if (this->CheckTransferFunctionString(v)) {
            this->val = v;
            this->hash = std::hash<std::string>()(this->val);
            this->indicateChange();
            if (setDirty)
                this->setDirty();
        }
    }
}


std::string TransferFunctionParam::ValueString(void) const {
    return this->val;
}


bool megamol::core::param::TransferFunctionParam::IgnoreProjectRange(const std::string& in_tfs) {

    bool ignore_project_range = true;
    try {
        if (!in_tfs.empty()) {
            ignore_project_range = false;
            nlohmann::json json = nlohmann::json::parse(in_tfs);
            megamol::core::utility::get_json_value<bool>(json, {"IgnoreProjectRange"}, &ignore_project_range);
        }
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON Error - Unable to read flag 'IgnoreProjectRange' from transfer function JSON string. [%s, %s, line "
            "%d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        ;
    }
    return ignore_project_range;
}


bool TransferFunctionParam::GetParsedTransferFunctionData(const std::string& in_tfs, NodeVector_t& out_nodes,
    InterpolationMode& out_interpolmode, unsigned int& out_texsize, std::array<float, 2>& out_range) {

    // DEFAULT for empty transfer function value string:
    NodeVector_t tmp_nodes;
    tmp_nodes.push_back({1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.05f});
    tmp_nodes.push_back({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.05f});
    std::string tmp_interpolmode_str;
    InterpolationMode tmp_interpolmode = InterpolationMode::LINEAR;
    unsigned int tmp_texsize = 256;
    std::array<float, 2> tmp_range = {0.0f, 1.0f};

    if (!in_tfs.empty()) {

        // Check for valid JSON string
        if (!TransferFunctionParam::CheckTransferFunctionString(in_tfs)) {
            return false;
        }
        try {
            nlohmann::json json = nlohmann::json::parse(in_tfs);

            megamol::core::utility::get_json_value<unsigned int>(json, {"TextureSize"}, &tmp_texsize);
            megamol::core::utility::get_json_value<float>(json, {"ValueRange"}, tmp_range.data(), 2);
            megamol::core::utility::get_json_value<std::string>(json, {"Interpolation"}, &tmp_interpolmode_str);
            if (tmp_interpolmode_str == "LINEAR") {
                tmp_interpolmode = InterpolationMode::LINEAR;
            } else if (tmp_interpolmode_str == "GAUSS") {
                tmp_interpolmode = InterpolationMode::GAUSS;
            }
            auto node_count = json.at("Nodes").size(); // unknown size
            tmp_nodes.resize(node_count);
            for (size_t i = 0; i < node_count; ++i) {
                json.at("Nodes")[i].get_to(tmp_nodes[i]);
            }
        } catch (...) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unknown Error - Unable to read transfer function from JSON string. [%s, %s, line %d]\n", __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }
    }

    if (!TransferFunctionParam::CheckTransferFunctionData(tmp_nodes, tmp_interpolmode, tmp_texsize, tmp_range)) {
        return false;
    }

    out_nodes = tmp_nodes;
    out_interpolmode = tmp_interpolmode;
    out_texsize = tmp_texsize;
    out_range = tmp_range;

    return true;
}


bool TransferFunctionParam::GetDumpedTransferFunction(std::string& out_tfs, const NodeVector_t& in_nodes,
    InterpolationMode in_interpolmode, unsigned int in_texsize, const std::array<float, 2>& in_range,
    bool in_ignore_project_range) {

    try {
        nlohmann::json json;

        if (!TransferFunctionParam::CheckTransferFunctionData(in_nodes, in_interpolmode, in_texsize, in_range)) {
            return false;
        }

        std::string interpolation_str;
        switch (in_interpolmode) {
        case (InterpolationMode::LINEAR):
            interpolation_str = "LINEAR";
            break;
        case (InterpolationMode::GAUSS):
            interpolation_str = "GAUSS";
            break;
        }

        json["Interpolation"] = interpolation_str;
        json["TextureSize"] = in_texsize;
        json["Nodes"] = in_nodes;
        json["ValueRange"] = in_range;
        json["IgnoreProjectRange"] = in_ignore_project_range;

        out_tfs = json.dump(2); // Dump with indent of 2 spaces and new lines.
    } catch (nlohmann::json::type_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::invalid_iterator& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::out_of_range& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::other_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unknown Error - Unable to write transfer function to JSON string. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool TransferFunctionParam::CheckTransferFunctionData(const NodeVector_t& nodes, InterpolationMode interpolmode,
    unsigned int texsize, const std::array<float, 2>& range) {

    bool check = true;

    // Texture Size
    if (texsize < 1) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Texture size should be greater than 0. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        check = false;
    }

    // Nodes
    float last_value = -1.0f;
    for (auto& a : nodes) {
        for (int i = 0; i < 5; ++i) {
            if (a[i] < 0.0f) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Values must be greater than or equal to 0. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                check = false;
            } else if (a[i] > 1.0f) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Values must be less than or equal to 1. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                check = false;
            }
        }
        if (last_value > a[4]) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Values should be sorted from 0 to 1. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        } else {
            last_value = a[4];
        }
        if (a[5] <= 0.0f) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Sigma value must be greater than 0. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            check = false;
        }
    }

    return check;
}


bool TransferFunctionParam::CheckTransferFunctionString(const std::string& tfs) {

    bool check = true;
    if (!tfs.empty()) {

        try {
            nlohmann::json json;
            json = nlohmann::json::parse(tfs);

            // Check for valid JSON object
            if (!json.is_object()) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Given string is no valid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                return false;
            }

            // Check texture size
            if (!json.at("TextureSize").is_number_integer()) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Couldn't read 'TextureSize' as integer value. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                    __LINE__);
                check = false;
            }

            // Check interpolation mode
            if (json.at("Interpolation").is_string()) {
                std::string tmp_str;
                json.at("Interpolation").get_to(tmp_str);
                if ((tmp_str != "LINEAR") && (tmp_str != "GAUSS")) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Couldn't find 'Interpolation' mode. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                    check = false;
                }
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Couldn't read 'Interpolation' as string value. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                    __LINE__);
                check = false;
            }

            // Check transfer function node data
            if (json.at("Nodes").is_array()) {
                unsigned int tmp_size = (unsigned int)json.at("Nodes").size();
                for (unsigned int i = 0; i < tmp_size; ++i) {
                    if (!json.at("Nodes")[i].is_array()) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "Entries of 'Nodes' should be arrays. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                            __LINE__);
                        check = false;
                    } else {
                        if (json.at("Nodes")[i].size() != TFP_VAL_CNT) {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "Entries of 'Nodes' should be arrays of size %d. [%s, %s, line %d]\n", TFP_VAL_CNT,
                                __FILE__, __FUNCTION__, __LINE__);
                            check = false;
                        } else {
                            for (unsigned int k = 0; k < TFP_VAL_CNT; ++k) {
                                if (!json.at("Nodes")[i][k].is_number()) {
                                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                                        "Values in 'Nodes' arrays should be numbers. [%s, %s, line %d]\n", __FILE__,
                                        __FUNCTION__, __LINE__);
                                    check = false;
                                }
                            }
                        }
                    }
                }
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Couldn't read 'Nodes' as array. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                check = false;
            }

            // Check value range
            if (json.at("ValueRange").is_array()) {
                unsigned int tmp_size = (unsigned int)json.at("ValueRange").size();
                if (tmp_size != 2) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "There should be at two entries in 'ValueRange' array. [%s, %s, line %d]\n", __FILE__,
                        __FUNCTION__, __LINE__);
                    check = false;
                }
                for (unsigned int i = 0; i < tmp_size; ++i) {
                    if (!json.at("ValueRange")[i].is_number()) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "Values in 'ValueRange' array should be numbers. [%s, %s, line %d]\n", __FILE__,
                            __FUNCTION__, __LINE__);
                        check = false;
                    }
                }
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Couldn't read 'ValueRange' as array. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                check = false;
            }
        } catch (nlohmann::json::type_error& e) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        } catch (nlohmann::json::invalid_iterator& e) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        } catch (nlohmann::json::out_of_range& e) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        } catch (nlohmann::json::other_error& e) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        } catch (...) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unknown Error - Unable to parse JSON string. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    }

    return check;
}


std::vector<float> TransferFunctionParam::LinearInterpolation(unsigned int in_texsize, const NodeVector_t& in_nodes) {

    std::vector<float> out_texdata;
    if ((in_texsize == 0) || (in_nodes.size() < 2)) {
        return out_texdata;
    }
    out_texdata.resize(4 * in_texsize);
    out_texdata.assign(out_texdata.size(), 0.0f);

    // Idea: Loop over all pixels of TF texture, find node left and right of texel position and linear interpolate
    // node values.
    int right_node_idx = 0;
    for (int i = 0; i < in_texsize; i++) {
        float px_pos = static_cast<float>(i) / static_cast<float>(in_texsize - 1);
        while (in_nodes[right_node_idx][4] <= px_pos && right_node_idx < in_nodes.size() - 1) {
            right_node_idx++;
        }
        const NodeData_t& cx1 = in_nodes[std::max(0, right_node_idx - 1)];
        const NodeData_t& cx2 = in_nodes[right_node_idx];
        const float dist = cx2[4] - cx1[4];
        if (dist > 0.0f) {
            const float al = std::clamp((px_pos - cx1[4]) / dist, 0.0f, 1.0f);
            const float be = 1.0f - al;
            out_texdata[i * 4] = cx1[0] * be + cx2[0] * al;
            out_texdata[i * 4 + 1] = cx1[1] * be + cx2[1] * al;
            out_texdata[i * 4 + 2] = cx1[2] * be + cx2[2] * al;
            out_texdata[i * 4 + 3] = cx1[3] * be + cx2[3] * al;
        } else {
            out_texdata[i * 4] = cx1[0];
            out_texdata[i * 4 + 1] = cx1[1];
            out_texdata[i * 4 + 2] = cx1[2];
            out_texdata[i * 4 + 3] = cx1[3];
        }
    }
    return out_texdata;
}


std::vector<float> TransferFunctionParam::GaussInterpolation(unsigned int in_texsize, const NodeVector_t& in_nodes) {

    std::vector<float> out_texdata;
    if ((in_texsize == 0) || (in_nodes.size() < 1)) {
        return out_texdata;
    }
    out_texdata.resize(4 * in_texsize);
    out_texdata.assign(out_texdata.size(), 0.0f);

    const float texel_size = 1.0f / static_cast<float>(in_texsize);

    float gb, gc;
    float r, g, b, a;
    size_t data_cnt = in_nodes.size();
    for (size_t i = 0; i < data_cnt; i++) {
        gb = in_nodes[i][4];
        gc = in_nodes[i][5];

        for (unsigned int t = 0; t < in_texsize; ++t) {
            float norm_x = texel_size * (static_cast<float>(t) + 0.5f);

            r = param::TransferFunctionParam::gauss(norm_x, in_nodes[i][0], gb, gc);
            g = param::TransferFunctionParam::gauss(norm_x, in_nodes[i][1], gb, gc);
            b = param::TransferFunctionParam::gauss(norm_x, in_nodes[i][2], gb, gc);
            a = param::TransferFunctionParam::gauss(norm_x, in_nodes[i][3], gb, gc);

            // Max
            out_texdata[t * 4] = std::max(r, out_texdata[t * 4]);
            out_texdata[t * 4 + 1] = std::max(g, out_texdata[t * 4 + 1]);
            out_texdata[t * 4 + 2] = std::max(b, out_texdata[t * 4 + 2]);
            out_texdata[t * 4 + 3] = std::max(a, out_texdata[t * 4 + 3]);
        }
    }
    return out_texdata;
}


bool megamol::core::param::TransferFunctionParam::GetTextureData(
    const std::string& in_tfs, std::vector<float>& out_tex_data, int& out_width, int& out_height) {

    out_tex_data.clear();
    unsigned int tmp_texture_size = 0;
    std::array<float, 2> tmp_range = {0.0f, 0.0f};
    megamol::core::param::TransferFunctionParam::NodeVector_t tmp_nodes;
    megamol::core::param::TransferFunctionParam::InterpolationMode tmp_mode;

    if (!megamol::core::param::TransferFunctionParam::GetParsedTransferFunctionData(
            in_tfs, tmp_nodes, tmp_mode, tmp_texture_size, tmp_range)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("Could not parse transfer function.");
        return false;
    }
    if (tmp_mode == param::TransferFunctionParam::InterpolationMode::LINEAR) {
        out_tex_data = param::TransferFunctionParam::LinearInterpolation(tmp_texture_size, tmp_nodes);
    } else if (tmp_mode == param::TransferFunctionParam::InterpolationMode::GAUSS) {
        out_tex_data = param::TransferFunctionParam::GaussInterpolation(tmp_texture_size, tmp_nodes);
    }
    out_width = static_cast<int>(tmp_texture_size);
    out_height = 1;

    return true;
}
