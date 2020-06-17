#pragma once

#include <vector>

namespace megamol {
namespace remote {
enum MessageType : unsigned char { NULL_MSG = 0u, PRJ_FILE_MSG, CAM_UPD_MSG, PARAM_UPD_MSG, HEAD_DISC_MSG };

using Message_t = std::vector<char>;

struct Message {
    MessageType type;
    uint64_t size;
    uint64_t id;
    Message_t msg_body;
};

constexpr size_t MessageTypeSize = sizeof(MessageType);
constexpr size_t MessageSizeSize = sizeof(uint64_t);
constexpr size_t MessageIDSize = sizeof(uint64_t);
constexpr size_t MessageHeaderSize = MessageIDSize + MessageTypeSize + MessageSizeSize;

} // end namespace remote
} // end namespace megamol
