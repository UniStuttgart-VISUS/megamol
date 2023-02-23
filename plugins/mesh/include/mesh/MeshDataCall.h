/*
 * MeshDataCall.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetDataCall.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace megamol::mesh {
/**
 * Call for transporting scalar data, attached with a transfer function, in an ready-to-use fashion (for OpenGL).
 *
 * @author Alexander Straub
 */
class MeshDataCall : public core::AbstractGetDataCall {
public:
    typedef core::factories::CallAutoDescription<MeshDataCall> mesh_data_description;

    /** Struct to store data and relevant information for visualization */
    struct data_set {
        std::string transfer_function;
        bool transfer_function_dirty;

        float min_value, max_value;

        std::shared_ptr<std::vector<float>> data;
    };

    /**
     * Human-readable class name
     */
    static const char* ClassName() {
        return "MeshDataCall";
    }

    /**
     * Human-readable class description
     */
    static const char* Description() {
        return "Call transporting data stored in a mesh";
    }

    /**
     * Number of available functions
     */
    static unsigned int FunctionCount() {
        return 2;
    }

    /**
     * Names of available functions
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "get_data";
        case 1:
            return "get_extent";
        }

        return nullptr;
    }

    /**
     * Set the data for a given name
     *
     * @param name Name of the data set
     * @param data Data set
     */
    void set_data(const std::string& name, std::shared_ptr<data_set> data = nullptr);

    /**
     * Get the data, as indicated by the name
     *
     * @param name Name of the data set
     *
     * @return Data set, or nullptr if it does not exist
     */
    std::shared_ptr<data_set> get_data(const std::string& name) const;

    /**
     * Get data set names
     *
     * @return Names of all available data sets
     */
    std::vector<std::string> get_data_sets() const;

    /**
     * Set the validity mask for a given name
     *
     * @param name Name of the mask
     * @param data Validity mask
     */
    void set_mask(const std::string& name, std::shared_ptr<std::vector<float>> mask = nullptr);

    /**
     * Get the validity mask, as indicated by the name
     *
     * @param name Name of the validity mask
     *
     * @return Validity mask, or nullptr if it does not exist
     */
    std::shared_ptr<std::vector<float>> get_mask(const std::string& name) const;

    /**
     * Get validity mask names
     *
     * @return Names of all available validity masks
     */
    std::vector<std::string> get_masks() const;

protected:
    /** Store data sets with their name */
    std::map<std::string, std::shared_ptr<data_set>> data_sets;

    /** Store validity masks with their name */
    std::map<std::string, std::shared_ptr<std::vector<float>>> masks;
};
} // namespace megamol::mesh
