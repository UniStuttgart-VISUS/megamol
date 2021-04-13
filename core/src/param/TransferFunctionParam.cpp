/*
 * TransferFunctionParam.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/TransferFunctionParam.h"


using namespace megamol::core::param;


TransferFunctionParam::TransferFunctionParam(const std::string& initVal)
        : AbstractParam(), val(""), hash(0) {

    if (this->CheckTransferFunctionString(initVal)) {
        this->val = initVal;
        this->hash = std::hash<std::string>()(this->val);
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[TransferFunctionParam] No valid parameter value for constructor given.");
    }
    this->InitPresentation(AbstractParamPresentation::ParamType::TRANSFERFUNCTION);
}


TransferFunctionParam::TransferFunctionParam(const char* initVal)
        : AbstractParam(), val(""), hash(0) {

    if (this->CheckTransferFunctionString(std::string(initVal))) {
        this->val = std::string(initVal);
        this->hash = std::hash<std::string>()(this->val);
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[TransferFunctionParam] No valid parameter value for constructor given.");
    }
    this->InitPresentation(AbstractParamPresentation::ParamType::TRANSFERFUNCTION);
}


TransferFunctionParam::TransferFunctionParam(const vislib::StringA& initVal)
        : AbstractParam(), val(""), hash(0) {

    if (this->CheckTransferFunctionString(std::string(initVal.PeekBuffer()))) {
        this->val = std::string(initVal.PeekBuffer());
        this->hash = std::hash<std::string>()(this->val);
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[TransferFunctionParam] No valid parameter value for constructor given.");
    }
    this->InitPresentation(AbstractParamPresentation::ParamType::TRANSFERFUNCTION);
}


TransferFunctionParam::~TransferFunctionParam(void) {}


void TransferFunctionParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(6);
    memcpy(outDef.AsAt<char>(0), "MMTFFC", 6);
}


bool TransferFunctionParam::ParseValue(vislib::TString const& v) {

    try {
        this->SetValue(std::string(v.PeekBuffer()));
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
            if (setDirty) this->setDirty();
        }
    }
}


vislib::TString TransferFunctionParam::ValueString(void) const {

    return vislib::TString(this->val.c_str());
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
            "[IgnoreProjectRange] JSON Error - Unable to read flag 'IgnoreProjectRange' from transfer function JSON string.");
    }
    return ignore_project_range;
}


bool TransferFunctionParam::GetParsedTransferFunctionData(const std::string& in_tfs, TFNodeType& out_nodes,
    InterpolationMode& out_interpolmode, unsigned int& out_texsize, std::array<float, 2>& out_range) {

    TFNodeType tmp_nodes;
    std::array<float, TFP_VAL_CNT> zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.05f};
    std::array<float, TFP_VAL_CNT> one = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.05f};
    tmp_nodes.emplace_back(zero);
    tmp_nodes.emplace_back(one);
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
        }
        catch (...) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ParseTransferFunction] Unknown Error - Unable to read transfer function from JSON string.");
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


bool TransferFunctionParam::GetDumpedTransferFunction(std::string& out_tfs, const TFNodeType& in_nodes,
    const InterpolationMode in_interpolmode, const unsigned int in_texsize, std::array<float, 2> in_range,
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
    }
    catch (nlohmann::json::type_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::invalid_iterator& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::out_of_range& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::other_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[DumpTransferFunction] Unknown Error - Unable to write transfer function to JSON string.");
        return false;
    }

    return true;
}


bool TransferFunctionParam::CheckTransferFunctionData(const TFNodeType& nodes, const InterpolationMode interpolmode,
    const unsigned int texsize, const std::array<float, 2> range) {

    bool check = true;

    // Range
    if (range[0] == range[1]) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[CheckTransferFunctionData] Range values should not be equal.");
        check = false;
    }

    // Texture Size
    if (texsize < 1) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[CheckTransferFunctionData] Texture size should be greater than 0.");
        check = false;
    }

    // Dat Size
    if (nodes.size() < 2) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[CheckTransferFunctionData] There should be at least two nodes.");
        check = false;
    }

    // Nodes
    float last_value = -1.0f;
    for (auto& a : nodes) {
        for (int i = 0; i < 5; ++i) {
            if (a[i] < 0.0f) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionData] Values must be greater than or equal to 0.");
                check = false;
            } else if (a[i] > 1.0f) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionData] Values must be less than or equal to 1.");
                check = false;
            }
        }
        if (last_value > a[4]) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[TransferFunction] 'Values' should be sorted from 0 to 1.");
            return false;
        } else {
            last_value = a[4];
        }
        if (a[5] <= 0.0f) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("[CheckTransferFunctionData] Sigma value must be greater than 0.");
            check = false;
        }
    }
    if (nodes.front()[4] != 0.0f) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[CheckTransferFunctionData] First node should have 'Value' = 0.");
        check = false;
    }
    if (nodes.back()[4] != 1.0f) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[CheckTransferFunctionData] Last node should have 'Value' = 1.");
        check = false;
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
                    "[CheckTransferFunctionString] Given string is no valid JSON object.");
                return false;
            }

            // Check texture size
            if (!json.at("TextureSize").is_number_integer()) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionString] Couldn't read 'TextureSize' as integer value.");
                check = false;
            }

            // Check interpolation mode
            if (json.at("Interpolation").is_string()) {
                std::string tmp_str;
                json.at("Interpolation").get_to(tmp_str);
                if ((tmp_str != "LINEAR") && (tmp_str != "GAUSS")) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[CheckTransferFunctionString] Couldn't find 'Interpolation' mode.");
                    check = false;
                }
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionString] Couldn't read 'Interpolation' as string value.");
                check = false;
            }

            // Check transfer function node data
            if (json.at("Nodes").is_array()) {
                unsigned int tmp_size = (unsigned int)json.at("Nodes").size();
                if (tmp_size < 2) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[CheckTransferFunctionString] There should be at least two entries in 'Nodes' array.");
                    check = false;
                }
                for (unsigned int i = 0; i < tmp_size; ++i) {
                    if (!json.at("Nodes")[i].is_array()) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[CheckTransferFunctionString] Entries of 'Nodes' should be arrays.");
                        check = false;
                    } else {
                        if (json.at("Nodes")[i].size() != TFP_VAL_CNT) {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[CheckTransferFunctionString] Entries of 'Nodes' should be arrays of size %d.",
                                TFP_VAL_CNT);
                            check = false;
                        } else {
                            for (unsigned int k = 0; k < TFP_VAL_CNT; ++k) {
                                if (!json.at("Nodes")[i][k].is_number()) {
                                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                                        "[CheckTransferFunctionString] Values in 'Nodes' arrays should be numbers.");
                                    check = false;
                                }
                            }
                        }
                    }
                }
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError("[CheckTransferFunctionString] Couldn't read 'Nodes' as array.");
                check = false;
            }

            // Check value range
            if (json.at("ValueRange").is_array()) {
                unsigned int tmp_size = (unsigned int)json.at("ValueRange").size();
                if (tmp_size != 2) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[CheckTransferFunctionString] There should be at two entries in 'ValueRange' array.");
                    check = false;
                }
                for (unsigned int i = 0; i < tmp_size; ++i) {
                    if (!json.at("ValueRange")[i].is_number()) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[CheckTransferFunctionString] Values in 'ValueRange' array should be numbers.");
                        check = false;
                    }
                }
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionString] Couldn't read 'ValueRange' as array.");
                check = false;
            }
        }
        catch (nlohmann::json::type_error& e) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::invalid_iterator& e) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::out_of_range& e) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::other_error& e) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (...) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("[CheckTransferFunctionString] Unknown Error - Unable to parse JSON string.");
            return false;
        }
    }

    return check;
}


std::vector<float> TransferFunctionParam::LinearInterpolation(unsigned int in_texsize, const TFNodeType& in_nodes) {

    std::vector<float> out_texdata;
    if (in_texsize == 0) {
        return out_texdata;
    }
    out_texdata.resize(4 * in_texsize);

    std::array<float, TFP_VAL_CNT> cx1 = in_nodes[0];
    std::array<float, TFP_VAL_CNT> cx2 = in_nodes[0];
    int p1 = 0;
    int p2 = 0;
    size_t data_cnt = in_nodes.size();
    for (size_t i = 1; i < data_cnt; i++) {
        cx1 = cx2;
        p1 = p2;
        cx2 = in_nodes[i];
        assert(cx2[4] <= 1.0f + 1e-5f); // 1e-5f = vislib::math::FLOAT_EPSILON
        p2 = static_cast<int>(cx2[4] * static_cast<float>(in_texsize - 1));
        assert(p2 < static_cast<int>(in_texsize));
        assert(p2 >= p1);

        for (int p = p1; p <= p2; p++) {
            float al = static_cast<float>(p - p1) / static_cast<float>(p2 - p1);
            float be = 1.0f - al;

            out_texdata[p * 4] = cx1[0] * be + cx2[0] * al;
            out_texdata[p * 4 + 1] = cx1[1] * be + cx2[1] * al;
            out_texdata[p * 4 + 2] = cx1[2] * be + cx2[2] * al;
            out_texdata[p * 4 + 3] = cx1[3] * be + cx2[3] * al;
        }
    }
    return out_texdata;
}


std::vector<float> TransferFunctionParam::GaussInterpolation(unsigned int in_texsize, const TFNodeType& in_nodes) {

    std::vector<float> out_texdata;
    if (in_texsize == 0) {
        return out_texdata;
    }
    out_texdata.resize(4 * in_texsize);
    out_texdata.assign(out_texdata.size(), 0.0f);

    float gb, gc;
    float x;
    float r, g, b, a;
    size_t data_cnt = in_nodes.size();
    for (size_t i = 0; i < data_cnt; i++) {
        gb = in_nodes[i][4];
        gc = in_nodes[i][5];
        for (unsigned int t = 0; t < in_texsize; ++t) {
            x = (float)t / (float)in_texsize;
            r = param::TransferFunctionParam::gauss(x, in_nodes[i][0], gb, gc);
            g = param::TransferFunctionParam::gauss(x, in_nodes[i][1], gb, gc);
            b = param::TransferFunctionParam::gauss(x, in_nodes[i][2], gb, gc);
            a = param::TransferFunctionParam::gauss(x, in_nodes[i][3], gb, gc);

            // Max
            out_texdata[t * 4] = std::max(r, out_texdata[t * 4]);
            out_texdata[t * 4 + 1] = std::max(g, out_texdata[t * 4 + 1]);
            out_texdata[t * 4 + 2] = std::max(b, out_texdata[t * 4 + 2]);
            out_texdata[t * 4 + 3] = std::max(a, out_texdata[t * 4 + 3]);
        }
    }
    return out_texdata;
}


 bool megamol::core::param::TransferFunctionParam::GetTextureData(const std::string & in_tfs, std::vector<float>& out_tex_data, int & out_width, int & out_height)  {

    out_tex_data.clear();
    unsigned int tmp_texture_size;
    std::array<float, 2> tmp_range;
    megamol::core::param::TransferFunctionParam::TFNodeType tmp_nodes;
    megamol::core::param::TransferFunctionParam::InterpolationMode tmp_mode;

    if (!megamol::core::param::TransferFunctionParam::GetParsedTransferFunctionData(in_tfs, tmp_nodes, tmp_mode, tmp_texture_size, tmp_range)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("Could not parse transfer function.");
        return false;
    }
    if (tmp_mode == param::TransferFunctionParam::InterpolationMode::LINEAR) {
        out_tex_data = param::TransferFunctionParam::LinearInterpolation(tmp_texture_size, tmp_nodes);
    }
    else if (tmp_mode == param::TransferFunctionParam::InterpolationMode::GAUSS) {
        out_tex_data = param::TransferFunctionParam::GaussInterpolation(tmp_texture_size, tmp_nodes);
    }
    out_width = tmp_texture_size;
    out_height = 1;

    return true;
 }
