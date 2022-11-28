/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <nlohmann/json.hpp>

#include "mmcore/utility/log/Log.h"
#include "mmcore/view/Camera.h"

namespace megamol::core::view {

/**
 * Class for the serialization and deserialization of camera parameters
 */
class CameraSerializer {
public:
    /**
     * Constructor
     *
     * @param prettyMode If set to true this activates the prettier formatting for files on disk
     */
    CameraSerializer(bool prettyMode = false);

    /**
     * Destructor
     */
    virtual ~CameraSerializer();

    /*
     * Serializes a single camera into a json string
     *
     * @param cam The camera to serialize.
     * @return A json string containing the serialized camera information
     */
    std::string serialize(Camera const& cam) const;

    /**
     * Serializes a vector of cameras into a string containing a json array
     *
     * @param camVec The vector containing the camera descriptions
     * @return A json string containing the information of the serialized cameras
     */
    std::string serialize(std::vector<Camera> const& camVec) const;

    /**
     * Serializes an array of cameras into a string containing a json array.
     * This method exists for the special case of serializing the cameras of a View.
     * These contain an additional validity flag. Both input arrays have to have the same length.
     *
     * @param camVec The array containing the camera information
     * @param validityFlags The array containing the validity flags
     * @return A json string containing the information of the serialized cameras
     */
    template<size_t N>
    std::string serialize(std::array<Camera, N> const& camVec, const std::array<bool, N>& validityFlags) const;

    /**
     * Serializes an array of cameras into a string containing a json array.
     * This method exists for the special case of serializing the cameras of a View.
     * These contain an additional validity flag.
     *
     * @param camVec The array containing the camera information
     * @return A json string containing the information of the serialized cameras
     */
    template<size_t N>
    std::string serialize(std::array<std::pair<Camera, bool>, N> const& camVec) const;

    /**
     * Deserializes a text containing a single camera description.
     * Please only use this method when you know the text contains only a single camera
     *
     * @param outCamera Output parameter that will contain the new camera if this method was successful
     * @param text The input text to deserialize
     * @return True on success, false otherwise
     */
    bool deserialize(Camera& outCamera, std::string const text) const;

    /**
     * Deserializes multiple camera at once when given a text containing a vector of camera descriptions
     * The input vector will be cleared before writing to it
     *
     * @param outCamera Output parameter that will contain a vector of cameras if this method was successful
     * @param text The input text to deserialize
     * @return True on success, false otherwise.
     */
    bool deserialize(std::vector<Camera>& outCameras, std::string const text) const;

    /**
     * Deserializes a predetermined amount of cameras at once, also reading the additionally stored validity flags.
     * This method exists predominantly to deserialize previously stored checkpoints of a View.
     * It will only be successful if the text to deserialize contains at least the specified amount of camera
     * descriptions. If there are no validity flags present in the input data, a flag of 'true' will be assumed.
     *
     * @param outCameras Output parameter that will contain the deserialized cameras
     * @param outValidity Output parameter that will contain the additionally read validity flags
     * @param text The input text to deserialize
     * @return True on success, false otherwise
     */
    template<size_t N>
    bool deserialize(std::array<Camera, N>& outCameras, std::array<bool, N>& outValidity, std::string const text) const;

    /**
     * Deserializes a predetermined amount of cameras at once, also reading the additionally stored validity flags.
     * This method exists predominantly to deserialize previously stored checkpoints of a View.
     * If there are no validity flags present in the input data, a flag of 'true' will be assumed.
     *
     * @param outCameras Output parameter that will contain the deserialized cameras alongside with the validity flags
     * @param text The input text to deserialize
     * @return True on success, false otherwise
     */
    template<size_t N>
    bool deserialize(std::array<std::pair<Camera, bool>, N>& outCameras, std::string const text) const;

    /**
     * Enables or disables the pretty serialization mode.
     *
     * @param pretty True if the pretty mode should be enabled, false otherwise. Default value is 'true'.
     */
    void setPrettyMode(bool prettyMode = true);

private:
    /*
     * Adds a camera description to the given json object.
     *
     * @param outObj The json object that wil contain the camera description
     * @param cam The camera description that will be stored in the json object
     */
    void addCamToJsonObject(nlohmann::json& outObj, Camera const& cam) const;

    /**
     * Extracts camera information from a json object that shall contain the camera.
     *
     * @param cam The read camera object
     * @param val The json value the camera object is read from
     * @param True if the reading was successful and either a 'true' valid flag is present or none at all. False
     * otherwise.
     */
    bool getCamFromJsonObject(Camera& cam, nlohmann::json::value_type const& val) const;

    /** The indentation for the pretty mode */
    const int prettyIndent = 2;

    /** Flag indication if the pretty mode is enabled for this serializer */
    bool prettyMode;
};

} // namespace megamol::core::view

#include "mmcore/view/CameraSerializer.inl"
