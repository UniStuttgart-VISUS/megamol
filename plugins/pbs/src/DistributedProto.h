#pragma once

#include <vector>

namespace megamol {
namespace pbs {
enum MessageType : unsigned char { NULL_MSG = 0u, PRJ_FILE_MSG, CAM_UPD_MSG, PARAM_UPD_MSG };

using MsgBody_t = std::vector<std::byte>;

struct Message {
    MessageType type;
    uint64_t size;
    MsgBody_t msg;
};

using Message_t = Message;

using MessageList_t = std::vector<Message_t>;
} // end namespace pbs
} // end namespace megamol
