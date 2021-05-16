/*
 * ExtractPores.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mesh/MeshDataCall.h"

#include <array>
#include <memory>
#include <vector>

namespace megamol {
namespace flowvis {
    /**
     * Module for splitting a surface mesh into pores and throats.
     *
     * @author Alexander Straub
     */
    class ExtractPores : public core::Module {
    public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline const char* ClassName() {
            return "ExtractPores";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline const char* Description() {
            return "Split a surface mesh into volumes representing pores and throats";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static inline bool IsAvailable() {
            return true;
        }

        /**
         * Initialises a new instance.
         */
        ExtractPores();

        /**
         * Finalises an instance.
         */
        virtual ~ExtractPores();

    protected:
        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create() override;

        /**
         * Implementation of 'Release'.
         */
        virtual void release() override;

    private:
        /** Callbacks for performing the computation and exposing data */
        bool getMeshDataCallback(core::Call& call);
        bool getMeshMetaDataCallback(core::Call& call);

        bool getMeshDataDataCallback(core::Call& call);
        bool getMeshDataMetaDataCallback(core::Call& call);

        /** Function to start the computation */
        bool compute();

        /** The slots for requesting data from this module, i.e., lhs connection */
        megamol::core::CalleeSlot mesh_lhs_slot, mesh_data_lhs_slot;

        /** The slots for querying data, i.e., a rhs connection */
        megamol::core::CallerSlot mesh_rhs_slot;

        /** Input */
        struct input_t {
            std::shared_ptr<std::vector<float>> vertices;
            std::shared_ptr<std::vector<float>> normals;
            std::shared_ptr<std::vector<unsigned int>> indices;
        } input;

        SIZE_T input_hash;

        /** Output */
        struct output_t {
            std::shared_ptr<std::vector<float>> vertices;
            std::shared_ptr<std::vector<float>> normals;
            std::shared_ptr<std::vector<unsigned int>> indices;

            std::array<std::shared_ptr<mesh::MeshDataCall::data_set>, 5> datasets;
        } output;
    };
} // namespace flowvis
} // namespace megamol
