#include "socket.h"
#include "vislib/Exception.h"

#include <array>

using namespace megamol::nvpipe;

socket::socket() {
	this->intialized = false;
}

socket::~socket() {

}

void socket::init(std::string serverName, PCSTR port) {
	// Initialize Winsock
	this->connectSocket == INVALID_SOCKET;

	this->returnCode = ::WSAStartup(MAKEWORD(2, 2), &this->wsaData);
	if (this->returnCode != 0) {
		throw std::exception("WSAStartup failed\n");
	}

	ZeroMemory(&this->hints, sizeof(this->hints));
	this->hints.ai_family = AF_UNSPEC;
	this->hints.ai_socktype = SOCK_STREAM;
	this->hints.ai_protocol = IPPROTO_TCP;

	// Resolve the server address and port
	this->returnCode = ::getaddrinfo(serverName.c_str(), port, &this->hints, &this->result);
	if (this->returnCode != 0) {
		::WSACleanup();
		throw std::exception("getaddrinfo failed\n");
	}
	this->intialized = true;
}


void socket::connect() {
	// Attempt to connect to an address until one succeeds
	for (ptr = this->result; ptr != NULL; ptr = ptr->ai_next) {

		// Create a SOCKET for connecting to server
		this->connectSocket = ::socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
		if (this->connectSocket == INVALID_SOCKET) {
			::WSACleanup();
			throw std::exception("socket failed\n");
		}

		// Connect to server.
		this->returnCode = ::connect(this->connectSocket, ptr->ai_addr, (int)ptr->ai_addrlen);
		if (this->returnCode == SOCKET_ERROR) {
			::closesocket(this->connectSocket);
			this->connectSocket = INVALID_SOCKET;
			continue;
		}
		break;
	}

	::freeaddrinfo(this->result);

	if (this->connectSocket == INVALID_SOCKET) {
		::WSACleanup();
		throw std::exception("Unable to connect to server!\n");
	}
}

void socket::sendInitialBuffer(int left, int top, int right, int bottom, int width, int height) {
	std::array<int, 6> bounds;
	bounds[0] = left;
	bounds[1] = top;
	bounds[2] = right;
	bounds[3] = bottom;
	bounds[4] = width;
	bounds[5] = height;
	std::transform(bounds.begin(), bounds.end(), bounds.begin(), ::htonl);
	
	this->returnCode = ::send(this->connectSocket, (char*)bounds.data(), sizeof(int) * 6, 0);
	if (this->returnCode == SOCKET_ERROR) {
		::closesocket(this->connectSocket);
		::WSACleanup();
		throw std::exception("send failed\n");
	}
}


void socket::sendFrame(size_t numBytes, uint8_t* sendBuffer) {
	this->returnCode = ::send(this->connectSocket, (char*)&numBytes, sizeof(size_t), 0);
	if (this->returnCode == SOCKET_ERROR) {
		::closesocket(this->connectSocket);
		::WSACleanup();
		throw std::exception("send failed\n");
	}

	this->returnCode = ::send(this->connectSocket, (char*)sendBuffer, numBytes, 0);
	if (this->returnCode == SOCKET_ERROR) {
	    ::closesocket(this->connectSocket);
		::WSACleanup();
		throw std::exception("send failed\n");
	}
}

void socket::closeConnection() {
	// clean shutdown the connection since no more data will be sent
	if (this->returnCode == 0) {
		returnCode = ::shutdown(this->connectSocket, SD_SEND);
		if (returnCode == SOCKET_ERROR) {
			::closesocket(this->connectSocket);
			::WSACleanup();
			throw std::exception("shutdown failed\n");
		} else {
			//cleanup
			::closesocket(this->connectSocket);
			::WSACleanup();
		}
	}
}

bool socket::isInitialized() {
	return this->intialized;
}