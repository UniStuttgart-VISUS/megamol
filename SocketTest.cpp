/*
 * SocketTest.cpp
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#include "stdafx.h"
#include "SocketTest.h"
#include "vislib/Log.h"
#include "vislib/SocketException.h"
#include "vislib/IPEndPoint.h"
#include "vislib/DNS.h"
#include "vislib/Thread.h"

using namespace megamol;
using namespace megamol::protein;

//#define TIMEOUT 100 // normally can use vislib::net::Socket::TIMEOUT_INFINITE

/*
 * SocketTest::SocketTest
 */
SocketTest::SocketTest(void) : socketValidity(false) { 
    // Communication with MDDriver
    try {
        // try to start up socket
        vislib::net::Socket::Startup();
        // create socket
        this->socket.Create(vislib::net::Socket::FAMILY_INET, vislib::net::Socket::TYPE_STREAM, vislib::net::Socket::PROTOCOL_TCP);
    } catch( vislib::net::SocketException e) {
        vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_ERROR, "Socket Exception during startup/create: %s", e.GetMsgA() );
    }
}


/*
 * SocketTest::~SocketTest
 */
SocketTest::~SocketTest(void) {
    this->terminateRequested = true;
    this->release();
}

/*
 * SocketTest::OnThreadStarting
 */
void SocketTest::OnThreadStarting(void *config) {
    this->terminateRequested = false;
    // Initialize or Reinitialize the socket connection
    this->startSocket( this->port);
}


/*
 * SocketTest::Run
 */
DWORD SocketTest::Run(void *config) {
    using vislib::sys::Log;

    while (this->socketValidity == true) {

        // Pause request
        if (this->pauseRequested == true && terminateRequested == false) {
            this->sendPause();
            this->pauseRequested = false; // flag pause done
            this->paused = true; // flag currently paused
        }

        // Go request
        if (this->goRequested == true && terminateRequested == false) {
            this->sendGo();
            this->sendPause();
            this->sendGo(); // DEBUG the second pause and go call is solving a bug with MDDriver
            this->goRequested = false; // flag go done
            this->paused = false; // flag not currently paused
        }

        // Transfer rate request
        if (this->rateRequested != 0 && terminateRequested == false) {
            this->sendTransferRate();
            this->rateRequested = 0; // flag transfer rate done
        }

        // Forces send request
        if (this->forcesCount != 0 && terminateRequested == false) {
            this->sendForces();
            this->forcesCount = 0; // flag force send done
        }

        // Get data if simulation is running
        if (this->paused == false && terminateRequested == false) {
            if (this->getData() == false && terminateRequested == false) {
                // get data failed - either attempt a reset or terminate
                if (this->reset < RESET_ATTEMPTS) {
                    // attempt a reset if the number of reset attempts has not yet been reached
                    this->sendPause();
                    this->sendGo();
                    this->reset += 1; // mark that another reset attempt was made
                } else {
                    this->terminateRequested = true; // returned false despite reset attempts - terminate the thread
                }
            }
        } else if (this->paused == true && terminateRequested == false) {
            vislib::sys::Thread::Sleep(50); // if MDDriver is paused, don't allow the thread to eat up so much CPU time
        }

        // Terminate connection request
        if (this->terminateRequested == true) {
            // send a disconnect signal and close the connection if there's a valid socket
            this->header.type = this->byteSwap(MDD_DISCONNECT);
            this->header.length = 0;
            try {
                this->socket.Send( &this->header, sizeof(MDDHeader), TIMEOUT, 0, true);
                // end the connection
                this->socket.Close();
            } catch( vislib::net::SocketException e) {
                Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "Socket Exception during disconnect: %s", e.GetMsgA() );
            }
            break; // leave the loop
        }
    }

    return 0;
}


/* 
 * SocketTest::Terminate
 */
bool SocketTest::Terminate(void) {
    using vislib::sys::Log;

    // clear all the requests
    this->terminateRequested = false;

    return true;
}

/*
 * SocketTest::Initialize
 */
void SocketTest::Initialize(int inPort) {
    this->port = inPort;
}

/*
 * SocketTest::startSocket
 */
bool SocketTest::startSocket( int port) {
    using vislib::sys::Log;

    try {
		this->socket.Bind(port);
		this->socket.SetDebug(false);
		this->socket.SetNoDelay(true);
		this->socket.Listen();
    } catch( vislib::net::SocketException e) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "Socket Exception during startup: %s", e.GetMsgA() );
        return false;
    }

    socketValidity = true; // flag the socket as being functional
    return true;
}

/*
 * SocketTest::release
 */
void SocketTest::release(void) {
    socketValidity = false; // flag the socket as being non-functional
    try {
        vislib::net::Socket::Cleanup();
    } catch( vislib::net::SocketException e ) {
        vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_ERROR, "Socket Exception during cleanup: %s", e.GetMsgA() );
    }
}
