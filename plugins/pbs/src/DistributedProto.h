#pragma once

#include <vector>

namespace megamol {
namespace pbs {

enum MessageType : unsigned char { NULL_MSG = 0u, PRJ_FILE_MSG, CAM_UPD_MSG, PARAM_UPD_MSG };

} // end namespace pbs
} // end namespace megamol
