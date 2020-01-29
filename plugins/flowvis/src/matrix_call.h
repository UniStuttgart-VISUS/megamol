/*
 * matrix_call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "glm/mat4x4.hpp"

namespace megamol {
namespace flowvis {

class matrix_call : public core::AbstractGetDataCall {
public:
    typedef core::factories::CallAutoDescription<matrix_call> matrix_call_description;

    /**
     * Human-readable class name
     */
    static const char* ClassName() { return "matrix_call"; }

    /**
     * Human-readable class description
     */
    static const char* Description() { return "Call transporting a matrix"; }

    /**
     * Number of available functions
     */
    static unsigned int FunctionCount() { return 1; }

    /**
     * Names of available functions
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "get_data";
        }

        return nullptr;
    }

    /** Default construction and destruction */
    matrix_call() = default;
    virtual ~matrix_call() noexcept = default;

    /**
     * Set the matrix
     */
    void set_matrix(const glm::mat4& matrix);

    /**
     * Get the stored matrix
     */
    glm::mat4 get_matrix() const;

protected:
    /** The transported matrix */
    glm::mat4 matrix;
};

} // namespace flowvis
} // namespace megamol