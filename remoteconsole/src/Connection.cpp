#include "Connection.h"
#include "stdafx.h"

Connection::Connection(zmq::socket_t& socket, const int timeOut) : socket(socket), activeHost(), timeOut(timeOut) {}

Connection::~Connection() {}
