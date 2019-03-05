/*
 * CameraSerializer.inl
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

/*
 * megamol::core::nextgen::CameraSerializer::serialize
 */
template <size_t N>
std::string megamol::core::nextgen::CameraSerializer::serialize(
    const std::array<megamol::core::nextgen::Camera_2::minimal_state_type, N>& camVec,
    const std::array<bool, N>& validityFlags) const {

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

template <size_t N>
std::string megamol::core::nextgen::CameraSerializer::serialize(
    const std::array<std::pair<megamol::core::nextgen::Camera_2::minimal_state_type, bool>, N>& camVec) const {

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
 * megamol::core::nextgen::CameraSerializer::deserialize
 */
template <size_t N>
bool megamol::core::nextgen::CameraSerializer::deserialize(
    std::array<megamol::core::nextgen::Camera_2::minimal_state_type, N>& outCameras, std::array<bool, N>& outValidity,
    const std::string text) const {
    nlohmann::json obj = nlohmann::json::parse(text);
    if (!obj.is_array()) {
        vislib::sys::Log::DefaultLog.WriteError("The input text does not contain a json array");
        return false;
    }
    if (obj.size() > N) {
        vislib::sys::Log::DefaultLog.WriteError("The number of objects in the text is smaller than the array size");
        return false;
    }
    for (nlohmann::json::iterator it = obj.begin(); it != obj.end(); ++it) {
        size_t index = static_cast<size_t>(it - obj.begin());
        auto cur = *it;
        megamol::core::nextgen::Camera_2::minimal_state_type cam;
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
 * megamol::core::nextgen::CameraSerializer::deserialize
 */
template <size_t N>
bool megamol::core::nextgen::CameraSerializer::deserialize(
    std::array<std::pair<megamol::core::nextgen::Camera_2::minimal_state_type, bool>, N>& outCameras,
    const std::string text) const {
    nlohmann::json obj = nlohmann::json::parse(text);
    if (!obj.is_array()) {
        vislib::sys::Log::DefaultLog.WriteError("The input text does not contain a json array");
        return false;
    }
    if (obj.size() > N) {
        vislib::sys::Log::DefaultLog.WriteError("The number of objects in the text is smaller than the array size");
        return false;
    }
    for (nlohmann::json::iterator it = obj.begin(); it != obj.end(); ++it) {
        size_t index = static_cast<size_t>(it - obj.begin());
        auto cur = *it;
        megamol::core::nextgen::Camera_2::minimal_state_type cam;
        bool result = this->getCamFromJsonObject(cam, cur);
        if (!result) {
            cam = {}; // empty the cam if it is garbage
        }
        outCameras[index].first = cam;
        outCameras[index].second = result;
    }
    return true;
}
