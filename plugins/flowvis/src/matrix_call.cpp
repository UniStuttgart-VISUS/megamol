#include "stdafx.h"
#include "matrix_call.h"

#include "glm/mat4x4.hpp"

namespace megamol {
namespace flowvis {

void matrix_call::set_matrix(const glm::mat4& matrix) { this->matrix = matrix; }

glm::mat4 matrix_call::get_matrix() const { return this->matrix; }

} // namespace flowvis
} // namespace megamol