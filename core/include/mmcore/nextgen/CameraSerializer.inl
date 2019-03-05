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

    // TODO
    return false;
}

/*
 * megamol::core::nextgen::CameraSerializer::deserialize
 */
template <size_t N>
bool megamol::core::nextgen::CameraSerializer::deserialize(
    std::array<std::pair<megamol::core::nextgen::Camera_2::minimal_state_type, bool>, N>& outCameras,
    const std::string text) const {

    // TODO
    return false;
}
