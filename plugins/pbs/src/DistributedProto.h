#pragma once

#include <vector>

namespace megamol {
namespace pbs {
enum MessageType : unsigned char { NULL_MSG = 0u, PRJ_FILE_MSG, CAM_UPD_MSG, PARAM_UPD_MSG };

using Message_t = std::vector<char>;

struct Message {
    MessageType type;
    uint64_t size;
    Message_t msg_body;
};

constexpr size_t MessageTypeSize = sizeof(MessageType);
constexpr size_t MessageSizeSize = sizeof(uint64_t);
constexpr size_t MessageHeaderSize = MessageTypeSize + MessageSizeSize;

} // end namespace pbs
} // end namespace megamol
