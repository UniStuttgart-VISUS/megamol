/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

/*
 * megamol::core::view::CameraSerializer::serialize
 */
template<size_t N>
std::string megamol::core::view::CameraSerializer::serialize(
    std::array<Camera, N> const& camVec, const std::array<bool, N>& validityFlags) const {

    nlohmann::json out;
    for (size_t i = 0; i < N; i++) {
        const auto& cam = camVec[i];
        nlohmann::json c;
        this->addCamToJsonObject(c, cam);
        c["valid"] = validityFlags[i]; // additional validity flag
        out.push_back(c);
    }
    if (this->prettyMode) {
        return out.dump(this->prettyIndent);
    }
    return out.dump();
}

template<size_t N>
std::string megamol::core::view::CameraSerializer::serialize(
    std::array<std::pair<Camera, bool>, N> const& camVec) const {

    nlohmann::json out;
    for (const auto& obj : camVec) {
        nlohmann::json c;
        this->addCamToJsonObject(c, obj.first);
        c["valid"] = obj.second;
        out.push_back(c);
    }
    if (this->prettyMode) {
        return out.dump(this->prettyIndent);
    }
    return out.dump();
}

/*
 * megamol::core::view::CameraSerializer::deserialize
 */
template<size_t N>
bool megamol::core::view::CameraSerializer::deserialize(
    std::array<Camera, N>& outCameras, std::array<bool, N>& outValidity, std::string const text) const {
    nlohmann::json obj = nlohmann::json::parse(text);
    if (!obj.is_array()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("The input text does not contain a json array");
        return false;
    }
    if (obj.size() > N) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "The number of objects in the text is smaller than the array size");
        return false;
    }
    for (nlohmann::json::iterator it = obj.begin(); it != obj.end(); ++it) {
        size_t index = static_cast<size_t>(it - obj.begin());
        auto cur = *it;
        megamol::core::view::Camera cam;
        bool result = this->getCamFromJsonObject(cam, cur);
        if (!result) {
            cam = {}; // empty the cam if it is garbage
        }
        outCameras[index] = cam;
        outValidity[index] = result;
    }
    return true;
}

/*
 * megamol::core::view::CameraSerializer::deserialize
 */
template<size_t N>
bool megamol::core::view::CameraSerializer::deserialize(
    std::array<std::pair<Camera, bool>, N>& outCameras, std::string const text) const {
    nlohmann::json obj = nlohmann::json::parse(text);
    if (!obj.is_array()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("The input text does not contain a json array");
        return false;
    }
    if (obj.size() > N) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "The number of objects in the text is smaller than the array size");
        return false;
    }
    for (nlohmann::json::iterator it = obj.begin(); it != obj.end(); ++it) {
        size_t index = static_cast<size_t>(it - obj.begin());
        auto cur = *it;
        megamol::core::view::Camera cam;
        bool result = this->getCamFromJsonObject(cam, cur);
        if (!result) {
            cam = {}; // empty the cam if it is garbage
        }
        outCameras[index].first = cam;
        outCameras[index].second = result;
    }
    return true;
}
