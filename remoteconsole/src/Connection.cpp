#include "stdafx.h"
#include "Connection.h"

Connection::Connection(zmq::socket_t& socket) : socket(socket), activeHost() {
}

Connection::~Connection() {
}
