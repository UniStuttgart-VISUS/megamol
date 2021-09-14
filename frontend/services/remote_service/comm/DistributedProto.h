#pragma once

#include <vector>

namespace megamol {
namespace remote {
enum class MessageType : unsigned char { NULL_MSG = 0u, PRJ_FILE_MSG, CAM_UPD_MSG, PARAM_UPD_MSG, HEAD_DISC_MSG };

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

//Message_t prepare_null_msg() {
//    Message_t msg(MessageHeaderSize);
//    auto const type = MessageType::NULL_MSG;
//    uint64_t const size = 0;
//    auto const size_ptr = reinterpret_cast<char const*>(&size);
//    msg[0] = static_cast<unsigned char>(type);
//    std::copy(size_ptr, size_ptr + sizeof(uint64_t), msg.begin() + 1);
//    return msg;
//}
//
//MessageType get_msg_type(Message_t const& msg) {
//    return static_cast<MessageType>(*msg.begin());
//}
//
//MessageType get_msg_type(Message_t::const_iterator const& begin, Message_t::const_iterator const& end) {
//    return static_cast<MessageType>(*begin);
//}
//
//uint64_t get_msg_size(Message_t::const_iterator const& begin, Message_t::const_iterator const& end) {
//    uint64_t ret = 0;
//    if (std::distance(begin, end) > MessageHeaderSize) {
//        std::copy(begin + MessageTypeSize, begin + MessageHeaderSize, &ret);
//    }
//    return ret;
//}
//
//Message_t get_msg(uint64_t size, Message_t::const_iterator const& begin, Message_t::const_iterator const& end) {
//    Message_t msg;
//    if (std::distance(begin, end) < MessageHeaderSize + size) {
//        return msg;
//    }
//
//    msg.resize(size);
//    std::copy(begin + MessageHeaderSize, begin + MessageHeaderSize + size, msg.begin());
//
//    return msg;
//}
//
//Message_t::const_iterator progress_msg(
//    uint64_t size, Message_t::const_iterator const& begin, Message_t::const_iterator const& end) {
//    if (std::distance(begin, end) > MessageHeaderSize + size) {
//        return begin + MessageHeaderSize + size;
//    }
//
//    return end;
//}


} // end namespace remote
} // end namespace megamol
