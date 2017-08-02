#pragma once

#include <string>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#include <ws2tcpip.h>
#elif _UNIX
#include <sys/types.h>
#include <sys/socket.h>
#endif

namespace megamol {
namespace nvpipe {

class socket {
public:

	socket();
	~socket();
	void init(std::string serverName, PCSTR port);
	void connect();
	void sendInitialBuffer(int left, int top, int right, int bottom, int width, int height);
	void sendFrame(size_t numBytes, uint8_t* serverSendBuffer);
	void closeConnection();
	bool isInitialized();
	size_t send(const void* buffer, const size_t numBytes);
	size_t receive(void* outData, const size_t cntBytes);

private:

	int returnCode;
	bool intialized;

	//win
	WSADATA wsaData;
	SOCKET connectSocket;
	struct addrinfo *result = NULL, *ptr = NULL, hints;

	//unix
	struct sockaddr_in serv_addr;
	struct hostent *server;

}; /* ending class socket */
} /* ending namespace megamol */
} /* ending namespace nvpipe */