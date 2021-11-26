#include "mmcore/utility/ZMQContextUser.h"
#include "stdafx.h"

std::weak_ptr<megamol::core::utility::ZMQContextUser> megamol::core::utility::ZMQContextUser::inst;

megamol::core::utility::ZMQContextUser::ptr megamol::core::utility::ZMQContextUser::Instance() {
    ptr p = inst.lock();
    if (!p) {
        inst = p = ptr(new ZMQContextUser());
    }
    return p;
}

megamol::core::utility::ZMQContextUser::ZMQContextUser() : context(1) {}

megamol::core::utility::ZMQContextUser::~ZMQContextUser() {}
