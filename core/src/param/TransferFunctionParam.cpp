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
TransferFunctionParam::TransferFunctionParam(const std::string& initVal) : val(initVal) {
    ///TODO:  Verify format of string -> JSON ...
}


/**
 * TransferFunctionParam::TransferFunctionParam
 */
TransferFunctionParam::TransferFunctionParam(const char *initVal) : val(initVal) {
    ///TODO:  Verify format of string -> JSON ...
}


/**
 * TransferFunctionParam::TransferFunctionParam
 */
TransferFunctionParam::TransferFunctionParam(const vislib::StringA& initVal) : val(initVal.PeekBuffer()) {
    ///TODO:  Verify format of string -> JSON ...
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
    memcpy(outDef.AsAt<char>(0), "MMTFFNC", 6);
}


/**
 * TransferFunctionParam::ParseValue
 */
bool TransferFunctionParam::ParseValue(vislib::TString const& v) {

    try {
        if (this->CheckTransferFunctionString(std::string(v.PeekBuffer()))) {
            this->val = std::string(v.PeekBuffer());
            return true;
        }
    } catch (...) {
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
            if (setDirty) this->setDirty();
        }
    }
}


/**
 * TransferFunctionParam::ValueString
 */
vislib::TString TransferFunctionParam::ValueString(void) const {
    return vislib::TString(this->val.c_str());
}


/**
 * TransferFunctionParam::ParseTransferFunction
 */
bool TransferFunctionParam::ParseTransferFunction(const std::string &in_tfs, std::vector<std::array<float, 5>> &out_data, InterpolationMode &out_interpolmode, size_t &out_texsize) {

    std::vector<std::array<float, 5>> tmp_data;
    std::string tmp_interpolmode_str;
    InterpolationMode tmp_interpolmode;
    unsigned int tmp_texsize;

    if (!in_tfs.empty()) {

        // Check for valid JSON string
        if (!TransferFunctionParam::CheckTransferFunctionString(in_tfs)) {
            return false;
        }

        nlohmann::json json = nlohmann::json::parse(in_tfs);

        // Get texture size
        json.at("TextureSize").get_to(tmp_texsize);

        // Get interpolation method
        json.at("Interpolation").get_to(tmp_interpolmode_str);
        if (tmp_interpolmode_str == "LINEAR") {
            tmp_interpolmode = InterpolationMode::LINEAR;
        }
        else if (tmp_interpolmode_str == "GAUSS") {
            tmp_interpolmode = InterpolationMode::GAUSS;
        }

        // Get nodes data
        size_t tf_size = json.at("Nodes").size();
        tmp_data.resize(tf_size);
        for (size_t i = 0; i < tf_size; ++i) {
            json.at("Nodes")[i].get_to(tmp_data[i]);
        }
    }
    else { // Loading default values for empty transfer function
        tmp_data.clear();
        std::array<float, 5> zero = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        std::array<float, 5> one = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
        tmp_data.emplace_back(zero);
        tmp_data.emplace_back(one);
        tmp_interpolmode = InterpolationMode::LINEAR;
        tmp_texsize = 128;
    }

    if (!TransferFunctionParam::CheckTransferFunctionData(tmp_data, tmp_interpolmode, (size_t)tmp_texsize)) {
        return false;
    }
    out_data = tmp_data;
    out_interpolmode = tmp_interpolmode;
    out_texsize = (size_t)tmp_texsize;

    return true;
}


/**
 * TransferFunctionParam::DumpTransferFunction
 */
bool TransferFunctionParam::DumpTransferFunction(std::string &out_tfs, const std::vector<std::array<float, 5>> &in_data, const InterpolationMode in_interpolmode, const size_t in_texsize) {

    nlohmann::json json;

    if (!TransferFunctionParam::CheckTransferFunctionData(in_data, in_interpolmode, in_texsize)) {
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
    json["Nodes"] = in_data;

    out_tfs = json.dump(); // pass 'true' for pretty printing with newlines

    return true;
}


/**
 * TransferFunctionParam::CheckTransferFunctionData
 */
bool TransferFunctionParam::CheckTransferFunctionData(const std::vector<std::array<float, 5>> &data, const InterpolationMode interpolmode, const size_t texsize) {

    bool check = true;
    if (texsize == 0) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[CheckTransferFunctionData] Texture size should be greater than 0.");
        check = false;
    }
    if (data.size() < 2) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[CheckTransferFunctionData] There should be at least two nodes.");
        check = false;
    }
    for (auto& a : data) {
        for (int i = 0; i < 4; ++i) {
            if (a[i] < 0.0f) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionData] Color values must be greater than or equal to 0.");
                check = false;
            }
            else if (a[i] > 1.0f) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionData] Color values must be less than or equal to 1.");
                check = false;
            }
        }
    }
    if (data.front()[4] != 0.0f) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[CheckTransferFunctionData] First node should have value 0.");
        check = false;
    }
    if (data.back()[4] != 1.0f) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[CheckTransferFunctionData] Last node should have value 1.");
        check = false;
    }

    return check;
}


/**
 * TransferFunctionParam::CheckTransferFunctionData
 */
bool TransferFunctionParam::CheckTransferFunctionString(const std::string &tfs) {

    bool check = true;
    if (!tfs.empty()) {

        nlohmann::json json = nlohmann::json::parse(tfs);

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
        if (!json.at("Interpolation").is_string()) {
            std::string tmp_str;
            json.at("Interpolation").get_to(tmp_str);
            if ((tmp_str != "LINEAR") && (tmp_str != "GAUSS")) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionString] Couldn't find 'Interpolation' mode.");
                check = false;
            }
        }
        else {
            vislib::sys::Log::DefaultLog.WriteError(
                "[CheckTransferFunctionString] Couldn't read 'Interpolation' as string value.");
            check = false;
        }

        // Check transfer function node data
        if (json.at("Nodes").is_array()) {
            size_t tmp_size = json.at("Nodes").size();
            if (tmp_size < 2) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[CheckTransferFunctionString] There should be at least two entries in 'Nodes' array.");
                check = false;
            }
            for (size_t i = 0; i < tmp_size; ++i) {
                if (!json.at("Nodes")[i].is_array()) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "[CheckTransferFunctionString] Entries of 'Nodes' should be arrays.");
                    check = false;
                }
                else {
                    if (json.at("Nodes")[i].size() != 5) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "[CheckTransferFunctionString] Entries of 'Nodes' should be arrays of size 5.");
                        check = false;
                    }
                    else {
                        for (size_t k = 0; k < 5; ++k) {
                            if (!json.at("Nodes")[i][k].is_number_float()) {
                                vislib::sys::Log::DefaultLog.WriteError(
                                    "[CheckTransferFunctionString] Values in 'Nodes' arrays should be floating point numbers.");
                                check = false;
                           }
                        }
                    }

                }
            }
        }
        else {
            vislib::sys::Log::DefaultLog.WriteError(
                "[CheckTransferFunctionString] Couldn't read 'Nodes' as array.");
            check = false;
        }
    }

    return check;
}