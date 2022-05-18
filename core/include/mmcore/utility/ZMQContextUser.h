#pragma once

#include <memory>
#include <zmq.hpp>

namespace megamol {
namespace core {
namespace utility {

class ZMQContextUser {
public:
    typedef std::shared_ptr<ZMQContextUser> ptr;

    static ptr Instance();

    inline zmq::context_t& Context() {
        return context;
    }
    inline const zmq::context_t& Context() const {
        return context;
    }
    inline operator zmq::context_t&() {
        return context;
    }
    inline operator const zmq::context_t&() const {
        return context;
    }

    ~ZMQContextUser();

private:
    static std::weak_ptr<ZMQContextUser> inst;

    ZMQContextUser();

    zmq::context_t context;
};

} /* namespace utility */
} /* namespace core */
} /* namespace megamol */
