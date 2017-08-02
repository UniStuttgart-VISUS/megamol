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
	std::array<int32_t, 6> bounds;
	bounds[0] = left;
	bounds[1] = top;
	bounds[2] = right;
	bounds[3] = bottom;
	bounds[4] = width;
	bounds[5] = height;
	std::transform(bounds.begin(), bounds.end(), bounds.begin(), ::htonl);
	
	this->returnCode = this->send(bounds.data(), sizeof(int32_t) * 6);
	if (this->returnCode == SOCKET_ERROR) {
		::closesocket(this->connectSocket);
		::WSACleanup();
		throw std::exception("send failed\n");
	}
}


void socket::sendFrame(size_t numBytes, uint8_t* sendBuffer) {
	uint32_t nl_numBytes = ::htonl(numBytes);
	this->returnCode = this->send(&nl_numBytes, sizeof(nl_numBytes));
	if (this->returnCode == SOCKET_ERROR) {
		::closesocket(this->connectSocket);
		::WSACleanup();
		throw std::exception("send failed\n");
	}

	this->returnCode = this->send(sendBuffer, numBytes);
	if (this->returnCode == SOCKET_ERROR) {
	    ::closesocket(this->connectSocket);
		::WSACleanup();
		throw std::exception("send failed\n");
	}
}

size_t socket::send(const void* buffer, const size_t numBytes) {
	int lastSent = 0;
	size_t totalSent = 0;
	do {
		lastSent = ::send(this->connectSocket, static_cast<const char*>(buffer), static_cast<int>(numBytes - totalSent), 0);

		if ((lastSent >= 0) && (lastSent != SOCKET_ERROR)) {
			totalSent += static_cast<size_t>(lastSent);
		} else {
			throw std::exception("send failed\n");
		}
	} while ((totalSent < numBytes) && (lastSent > 0));
	return totalSent;
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
			this->intialized = false;
		}
	}
}


size_t socket::receive(void* outData, const size_t cntBytes) {
	size_t totalReceived = 0;
	int lastReceived = 0;

	do {
		lastReceived = ::recv(this->connectSocket, static_cast<char *>(outData) + totalReceived,
			static_cast<int>(cntBytes - totalReceived), 0);

		if ((lastReceived >= 0) && (lastReceived != SOCKET_ERROR)) {
			/* Successfully received new package. */
			totalReceived += static_cast<size_t>(lastReceived);
		} else {
			/* Communication failed. */
			throw std::exception("Receiving failed");
		}
	} while ((totalReceived < cntBytes) && (lastReceived > 0));

	return totalReceived;
}

bool socket::isInitialized() {
	return this->intialized;
}