/*
 * TransferFunctionParam.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/TransferFunctionParam.h"


using namespace megamol::core::param;


/**
 * TransferFunctionParam::TransferFunctionParam
 */
TransferFunctionParam::TransferFunctionParam(const std::string& initVal) : AbstractParam() {
    if (this->CheckTransferFunctionString(initVal)) {
        this->val = initVal;
        this->hash = std::hash<std::string>()(this->val);
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "[TransferFunctionParam] No valid parameter value for constructor given.");
    }
}


/**
 * TransferFunctionParam::TransferFunctionParam
 */
TransferFunctionParam::TransferFunctionParam(const char* initVal) : AbstractParam() {
    if (this->CheckTransferFunctionString(std::string(initVal))) {
        this->val = std::string(initVal);
        this->hash = std::hash<std::string>()(this->val);
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "[TransferFunctionParam] No valid parameter value for constructor given.");
    }
}


/**
 * TransferFunctionParam::TransferFunctionParam
 */
TransferFunctionParam::TransferFunctionParam(const vislib::StringA& initVal) : AbstractParam() {
    if (this->CheckTransferFunctionString(std::string(initVal.PeekBuffer()))) {
        this->val = std::string(initVal.PeekBuffer());
        this->hash = std::hash<std::string>()(this->val);
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "[TransferFunctionParam] No valid parameter value for constructor given.");
    }
}


/**
 * TransferFunctionParam::~TransferFunctionParam
 */
TransferFunctionParam::~TransferFunctionParam(void) {}


/**
 * TransferFunctionParam::Definition
 */
void TransferFunctionParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(6);
    memcpy(outDef.AsAt<char>(0), "MMTFFC", 6);
}


/**
 * TransferFunctionParam::ParseValue
 */
bool TransferFunctionParam::ParseValue(vislib::TString const& v) {

    if (this->CheckTransferFunctionString(std::string(v.PeekBuffer()))) {
        this->val = std::string(v.PeekBuffer());
        this->hash = std::hash<std::string>()(this->val);
        return true;
    }

    return false;
}


/**
 * TransferFunctionParam::SetValue
 */
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


/**
 * TransferFunctionParam::ValueString
 */
vislib::TString TransferFunctionParam::ValueString(void) const { return vislib::TString(this->val.c_str()); }


/**
 * TransferFunctionParam::TransferFunctionTexture
 */
bool TransferFunctionParam::TransferFunctionTexture(
    const std::string& in_tfs, std::vector<float>& out_nodes, UINT& out_texsize, std::array<float, 2>& out_range) {

    TFNodeType nodes;
    InterpolationMode mode;

    if (ParseTransferFunction(in_tfs, nodes, mode, out_texsize, out_range)) {
        if (mode == InterpolationMode::LINEAR) {
            LinearInterpolation(out_nodes, out_texsize, nodes);
        } else if (mode == InterpolationMode::GAUSS) {
            GaussInterpolation(out_nodes, out_texsize, nodes);
        } else {
            return false;
        }

        return true;
    }

    return false;
}


/**
 * TransferFunctionParam::ParseTransferFunction
 */
bool TransferFunctionParam::ParseTransferFunction(const std::string& in_tfs, TFNodeType& out_nodes,
    InterpolationMode& out_interpolmode, UINT& out_texsize, std::array<float, 2>& out_range) {

    TFNodeType tmp_nodes;
    std::string tmp_interpolmode_str;
    InterpolationMode tmp_interpolmode;
    UINT tmp_texsize;
    std::array<float, 2> tmp_range;

    if (!in_tfs.empty()) {

        // Check for valid JSON string
        if (!TransferFunctionParam::CheckTransferFunctionString(in_tfs)) {
            return false;
        }
        try {
            nlohmann::json json = nlohmann::json::parse(in_tfs);

            // Get texture size
            json.at("TextureSize").get_to(tmp_texsize);

            // Get interpolation method
            json.at("Interpolation").get_to(tmp_interpolmode_str);
            if (tmp_interpolmode_str == "LINEAR") {
                tmp_interpolmode = InterpolationMode::LINEAR;
            } else if (tmp_interpolmode_str == "GAUSS") {
                tmp_interpolmode = InterpolationMode::GAUSS;
            }

            // Get nodes data
            UINT tf_size = (UINT)json.at("Nodes").size();
            tmp_nodes.resize(tf_size);
            for (UINT i = 0; i < tf_size; ++i) {
                json.at("Nodes")[i].get_to(tmp_nodes[i]);
            }

            // Get value range
            json.at("ValueRange")[0].get_to(tmp_range[0]);
            json.at("ValueRange")[1].get_to(tmp_range[1]);

        }
        catch (nlohmann::json::type_error& e) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::exception& e) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::parse_error& e) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::invalid_iterator& e) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::out_of_range& e) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::other_error& e) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (...) {
            vislib::sys::Log::DefaultLog.WriteError(
                "[ParseTransferFunction] Unknown Error - Unable to read transfer function from JSON string.");
            return false;
        }
    } else { // Loading default values for empty transfer function
        tmp_nodes.clear();
        std::array<float, TFP_VAL_CNT> zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.05f};
        std::array<float, TFP_VAL_CNT> one = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.05f};
        tmp_nodes.emplace_back(zero);
        tmp_nodes.emplace_back(one);
        tmp_interpolmode = InterpolationMode::LINEAR;
        tmp_texsize = 256;
        tmp_range[0] = 0.0f;
        tmp_range[1] = 1.0f;
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


/**
 * TransferFunctionParam::DumpTransferFunction
 */
bool TransferFunctionParam::DumpTransferFunction(std::string& out_tfs, const TFNodeType& in_nodes,
    const InterpolationMode in_interpolmode, const UINT in_texsize, std::array<float, 2> in_range) {

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

        out_tfs = json.dump(2); // Dump with indent of 2 spaces and new lines.
    }
    catch (nlohmann::json::type_error& e) {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::exception& e) {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::parse_error& e) {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::invalid_iterator& e) {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::out_of_range& e) {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::other_error& e) {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[DumpTransferFunction] Unknown Error - Unable to write transfer function to JSON string.");
        return false;
    }

    return true;
}


/**
 * TransferFunctionParam::CheckTransferFunctionData
 */
bool TransferFunctionParam::CheckTransferFunctionData(const TFNodeType& nodes, const InterpolationMode interpolmode,
    const UINT texsize, const std::array<float, 2> range) {

    bool check = true;

    // Range
    if (range[0] == range[1]) {
        vislib::sys::Log::DefaultLog.WriteError("[CheckTransferFunctionData] Range values should not be equal.");
        check = false;
    }

    // Texture Size
    if (texsize < 1) {
        vislib::sys::Log::DefaultLog.WriteError("[CheckTransferFunctionData] Texture size should be greater than 0.");
        check = false;
    }

    // Dat Size
    if (nodes.size() < 2) {
        vislib::sys::Log::DefaultLog.WriteError("[CheckTransferFunctionData] There should be at least two nodes.");
        check = false;
    }

    // Nodes
    float last_value = -1.0f;
    for (auto& a : nodes) {
        for (int i = 0; i < 5; ++i) {
            if (a[i] < 0.0f) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionData] Values must be greater than or equal to 0.");
                check = false;
            } else if (a[i] > 1.0f) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionData] Values must be less than or equal to 1.");
                check = false;
            }
        }
        if (last_value > a[4]) {
            vislib::sys::Log::DefaultLog.WriteError(
                "[TransferFunction] 'Values' should be sorted from 0 to 1.");
            return false;
        } else {
            last_value = a[4];
        }
        if (a[5] <= 0.0f) {
            vislib::sys::Log::DefaultLog.WriteError("[CheckTransferFunctionData] Sigma value must be greater than 0.");
            check = false;
        }
    }
    if (nodes.front()[4] != 0.0f) {
        vislib::sys::Log::DefaultLog.WriteError("[CheckTransferFunctionData] First node should have 'Value' = 0.");
        check = false;
    }
    if (nodes.back()[4] != 1.0f) {
        vislib::sys::Log::DefaultLog.WriteError("[CheckTransferFunctionData] Last node should have 'Value' = 1.");
        check = false;
    }

    return check;
}


/**
 * TransferFunctionParam::CheckTransferFunctionData
 */
bool TransferFunctionParam::CheckTransferFunctionString(const std::string& tfs) {

    bool check = true;
    if (!tfs.empty()) {

        try {
            nlohmann::json json;
            json = nlohmann::json::parse(tfs);

            // Check for valid JSON object
            if (!json.is_object()) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionString] Given string is no valid JSON object.");
                return false;
            }

            // Check texture size
            if (!json.at("TextureSize").is_number_integer()) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionString] Couldn't read 'TextureSize' as integer value.");
                check = false;
            }

            // Check interpolation mode
            if (json.at("Interpolation").is_string()) {
                std::string tmp_str;
                json.at("Interpolation").get_to(tmp_str);
                if ((tmp_str != "LINEAR") && (tmp_str != "GAUSS")) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "[CheckTransferFunctionString] Couldn't find 'Interpolation' mode.");
                    check = false;
                }
            } else {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionString] Couldn't read 'Interpolation' as string value.");
                check = false;
            }

            // Check transfer function node data
            if (json.at("Nodes").is_array()) {
                UINT tmp_size = (UINT)json.at("Nodes").size();
                if (tmp_size < 2) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "[CheckTransferFunctionString] There should be at least two entries in 'Nodes' array.");
                    check = false;
                }
                for (UINT i = 0; i < tmp_size; ++i) {
                    if (!json.at("Nodes")[i].is_array()) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "[CheckTransferFunctionString] Entries of 'Nodes' should be arrays.");
                        check = false;
                    } else {
                        if (json.at("Nodes")[i].size() != TFP_VAL_CNT) {
                            vislib::sys::Log::DefaultLog.WriteError(
                                "[CheckTransferFunctionString] Entries of 'Nodes' should be arrays of size %d.",
                                TFP_VAL_CNT);
                            check = false;
                        } else {
                            for (UINT k = 0; k < TFP_VAL_CNT; ++k) {
                                if (!json.at("Nodes")[i][k].is_number()) {
                                    vislib::sys::Log::DefaultLog.WriteError(
                                        "[CheckTransferFunctionString] Values in 'Nodes' arrays should be numbers.");
                                    check = false;
                                }
                            }
                        }
                    }
                }
            } else {
                vislib::sys::Log::DefaultLog.WriteError("[CheckTransferFunctionString] Couldn't read 'Nodes' as array.");
                check = false;
            }

            // Check value range
            if (json.at("ValueRange").is_array()) {
                UINT tmp_size = (UINT)json.at("ValueRange").size();
                if (tmp_size != 2) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "[CheckTransferFunctionString] There should be at two entries in 'ValueRange' array.");
                    check = false;
                }
                for (UINT i = 0; i < tmp_size; ++i) {
                    if (!json.at("ValueRange")[i].is_number()) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "[CheckTransferFunctionString] Values in 'ValueRange' array should be numbers.");
                        check = false;
                    }
                }
            } else {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionString] Couldn't read 'ValueRange' as array.");
                check = false;
            }
        }
        catch (nlohmann::json::type_error& e) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::exception& e) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::parse_error& e) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::invalid_iterator& e) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::out_of_range& e) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::other_error& e) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (...) {
            vislib::sys::Log::DefaultLog.WriteError("[CheckTransferFunctionString] Unknown Error - Unable to parse JSON string.");
            return false;
        }
    }

    return check;
}


/*
 * TransferFunctionParam::LinearInterpolation
 */
void TransferFunctionParam::LinearInterpolation(
    std::vector<float>& out_texdata, unsigned int in_texsize, const TFNodeType& in_nodes) {

    if (in_texsize == 0) {
        return;
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
}


/*
 * TransferFunctionParam::GaussInterpolation
 */
void TransferFunctionParam::GaussInterpolation(
    std::vector<float>& out_texdata, unsigned int in_texsize, const TFNodeType& in_nodes) {

    if (in_texsize == 0) {
        return;
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
}
