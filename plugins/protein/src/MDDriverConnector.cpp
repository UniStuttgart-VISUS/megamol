/*
 * MDDriverConnector.cpp
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#include "MDDriverConnector.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/StringConverter.h"
#include "vislib/net/DNS.h"
#include "vislib/net/IPEndPoint.h"
#include "vislib/net/SocketException.h"
#include "vislib/sys/Thread.h"

using namespace megamol;
using namespace megamol::protein;

#define TIMEOUT 10000    // normally can use vislib::net::Socket::TIMEOUT_INFINITE
#define RESET_ATTEMPTS 1 // how many times MDDriver will be reset before deciding that it has failed

// TODO: Find a better way of hiding the error messages when the thread is terminated in the middle of a getData call

/*
 * MDDriverConnector::MDDriverConnector
 */
MDDriverConnector::MDDriverConnector(void)
        : socketValidity(false)
        , pauseRequested(false)
        , goRequested(false)
        , rateRequested(0)
        , zeroForce(false)
        , forcesCount(0)
        , terminateRequested(false)
        , paused(false)
        , reset(0)
        , atomCount(0) {

    // Communication with MDDriver
    try {
        // try to start up socket
        vislib::net::Socket::Startup();
        // create socket
        this->socket.Create(
            vislib::net::Socket::FAMILY_INET, vislib::net::Socket::TYPE_STREAM, vislib::net::Socket::PROTOCOL_TCP);
    } catch (vislib::net::SocketException e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Socket Exception during startup/create: %s", e.GetMsgA());
    }
}


/*
 * MDDriverConnector::~MDDriverConnector
 */
MDDriverConnector::~MDDriverConnector(void) {
    this->terminateRequested = true;
    this->release();
}

/*
 * MDDriverConnector::OnThreadStarting
 */
void MDDriverConnector::OnThreadStarting(void* config) {
    this->terminateRequested = false;
    // Initialize or Reinitialize the socket connection
    this->startSocket(this->host, this->port);
}


/*
 * MDDriverConnector::Run
 */
DWORD MDDriverConnector::Run(void* config) {
    using megamol::core::utility::log::Log;

    while (this->socketValidity == true) {

        // Pause request
        if (this->pauseRequested == true && terminateRequested == false) {
            this->sendPause();
            this->pauseRequested = false; // flag pause done
            this->paused = true;          // flag currently paused
        }

        // Go request
        if (this->goRequested == true && terminateRequested == false) {
            this->sendGo();
            this->sendPause();
            this->sendGo();            // DEBUG the second pause and go call is solving a bug with MDDriver
            this->goRequested = false; // flag go done
            this->paused = false;      // flag not currently paused
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
                this->socket.Send(&this->header, sizeof(MDDHeader), TIMEOUT, 0, true);
                // end the connection
                this->socket.Close();
            } catch (vislib::net::SocketException e) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Socket Exception during disconnect: %s", e.GetMsgA());
            }
            break; // leave the loop
        }
    }

    return 0;
}


/*
 * MDDriverConnector::Terminate
 */
bool MDDriverConnector::Terminate(void) {
    using megamol::core::utility::log::Log;

    // clear all the requests
    this->pauseRequested = false;
    this->goRequested = false;
    this->rateRequested = 0;
    this->forcesCount = 0;
    this->terminateRequested = false;
    this->reset = 0;
    this->atomCount = 0;

    return true;
}

/*
 * MDDriverConnector::Initialize
 */
void MDDriverConnector::Initialize(vislib::TString inHost, int inPort) {
    this->host = inHost;
    this->port = inPort;
}

/*
 * MDDriverConnector::RequestForces
 */
void MDDriverConnector::RequestForces(int count, const unsigned int* atomIDs, const float* forces) {
    if (count != 0) {
        if (this->forceIDs.TryLock()) {
            if (this->forceList.TryLock()) {
                // Copy the new forces into the forces arrays
                this->forceIDs.AssertCapacity(count);
                this->forceIDs.Clear();
                for (int iter = 0; iter < count; iter += 1) {
                    this->forceIDs.Add(atomIDs[iter]);
                }
                this->forceList.AssertCapacity(count * 3);
                this->forceList.Clear();
                for (int iter = 0; iter < count * 3; iter += 1) {
                    this->forceList.Add(forces[iter]);
                }
                this->forcesCount = count;
                this->zeroForce = true; // mark that the forces will need to be zeroed when they are removed
                this->forceList.Unlock();
            }
            this->forceIDs.Unlock();
        }
    } else if (count == 0 && this->zeroForce == true) {
        // create a 0 force to clear other forces (not sure this is necessary)
        if (this->forceIDs.TryLock()) {
            if (this->forceList.TryLock()) {
                this->forceIDs.AssertCapacity(1);
                this->forceIDs.Clear();
                // apply force to 0th atom
                this->forceIDs.Add(1);
                this->forceList.AssertCapacity(3);
                this->forceList.Clear();
                // apply 0 vector force
                this->forceList.Add(0.0f);
                this->forceList.Add(0.0f);
                this->forceList.Add(0.0f);
                this->forcesCount = 1;
                this->zeroForce = false; // mark that a zero force has been applied so it won't be applied again
                this->forceList.Unlock();
            }
            this->forceIDs.Unlock();
        } else {                   // if count == 0 and zeroForce == false
            this->forcesCount = 0; // don't apply any forces, not even a zero
        }
    }
}

/*
 * MDDriverConnector::GetCoordinates
 */
void MDDriverConnector::GetCoordinates(int count, float* atomPos) {
    using megamol::core::utility::log::Log;

    if (this->atomCount == count) {
        if (this->atomCoordinates.TryLock()) {
            memcpy(atomPos, atomCoordinates.PeekElements(), sizeof(float) * count * 3);
            this->atomCoordinates.Unlock();
        }
    } else if (this->atomCount == 0) {
        // do nothing - no data received yet, but it's coming, so no error message needed
    } else {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Atom count mismatch between plugin and MDDriver.");
        //printf("MDDriver Says: %d \n PDB Loader says: %d\n", this->atomCount, count);
    }
}

/*
 * MDDriverConnector::startSocket
 */
bool MDDriverConnector::startSocket(const vislib::TString& host, int port) {
    using megamol::core::utility::log::Log;
    // set default values
    header.type = 0;
    header.length = 0;

    try {
        // connect to MDDriver to allow real time data transfer
        this->socket.Connect(vislib::net::IPEndPoint::CreateIPv4(T2A(host), port));

        // handshake: receive a header
        if (this->socket.Receive(&this->header, sizeof(MDDHeader), TIMEOUT, 0, true) != sizeof(MDDHeader)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Handshake: did not receive full header to initiate.");
            return false;
        }
        // handshake: check that header type (byte swapped) is handshake (failure could indicate wrong endian)
        if (this->byteSwap(this->header.type) != MDD_HANDSHAKE) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Handshake: unexpected header type - expected %d, received %d",
                MDD_HANDSHAKE, this->byteSwap(this->header.type));
            return false;
        }
        // handshake: check that length (not byte swapped) gives the correct MDDriver version (failure could indicate wrong endian)
        if (this->header.length != MDD_VERSION) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Handshake: unexpected MDD version - expected %d, received %d",
                MDD_VERSION, this->header.length);
            return false;
        }
        // handshake: send a go signal to complete the handshake
        this->header.type = this->byteSwap(MDD_GO);
        this->header.length = 0;
        if (this->socket.Send(&this->header, sizeof(MDDHeader), TIMEOUT, 0, true) != sizeof(MDDHeader)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Handshake: did not send full header to initiate.");
            return false;
        }
    } catch (vislib::net::SocketException e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Socket Exception during connect/handshake: %s", e.GetMsgA());
        return false;
    }

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "MDDriver socket connection successfully started and configured.");
    socketValidity = true; // flag the socket as being functional
    return true;
}

/*
 * MDDriverConnector::release
 */
void MDDriverConnector::release(void) {
    socketValidity = false; // flag the socket as being non-functional
    try {
        vislib::net::Socket::Cleanup();
    } catch (vislib::net::SocketException e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Socket Exception during cleanup: %s", e.GetMsgA());
    }
}

/*
 * MDDriverConnector::getData
 */
bool MDDriverConnector::getData(void) {
    using megamol::core::utility::log::Log;

    bool retval = true;

    retval &= this->getHeader(); // see what kind of data is arriving

    if (this->header.type == MDD_DISCONNECT) {
        // MDDriver sent a disconnect signal - end the connection
        if (this->terminateRequested == false) {
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_INFO, "MDDriver disconnect signal received - closing socket connection.");
        }
        return false;

    } else if (this->header.type == MDD_COORDS) {
        // server is sending coordinate data - allocate space and take the data
        this->atomCoordinates.Lock(); // lock the data array for data transfer

        this->atomCount = header.length; // save the input data size
        this->atomCoordinates.AssertCapacity(
            this->header.length * 3); // coordinates should be able to hold x,y,z floats for all atoms
        try {
            this->socket.Receive(const_cast<float*>(this->atomCoordinates.PeekElements()),
                sizeof(float) * 3 * this->header.length, TIMEOUT, 0, true);
        } catch (vislib::net::SocketException e) {
            if (this->reset >= RESET_ATTEMPTS) {
                if (this->terminateRequested == false) {
                    // This branch will only occur if MDDriver has already reached its reset attempts this thread AND the thread is not being asked to terminate
                    // In other words, this branch only occurs if the thread thinks it should be running but MDDriver has failed for some reason.
                    // The return false here should trigger a terminate request for the thread.
                    Log::DefaultLog.WriteMsg(
                        Log::LEVEL_ERROR, "Socket Exception during atom coordinates receive: %s", e.GetMsgA());
                }
            }
            retval = false;
        }

        this->atomCoordinates.Unlock(); // unlock the data array

    } else if (this->header.type == MDD_ENERGIES) {
        // verify that the energies are coming by checking that the length is 1 (number assigned by MDDriver)
        if (this->header.length != 1) {
            if (this->terminateRequested == false) {
                Log::DefaultLog.WriteMsg(
                    Log::LEVEL_ERROR, "Energy table 'length' mismatch between plugin and MDDriver.");
            }
            retval = false;
        } else {
            // take the data
            try {
                this->socket.Receive(&this->energies, sizeof(MDDEnergies), TIMEOUT, 0, true);
            } catch (vislib::net::SocketException e) {
                if (this->reset >= RESET_ATTEMPTS) {
                    if (this->terminateRequested == false) {
                        // This branch will only occur if MDDriver has already reached its reset attempts this thread AND the thread is not being asked to terminate
                        // In other words, this branch only occurs if the thread thinks it should be running but MDDriver has failed for some reason.
                        // The return false here should trigger a terminate request for the thread.
                        Log::DefaultLog.WriteMsg(
                            Log::LEVEL_ERROR, "Socket Exception during energies receive: %s", e.GetMsgA());
                    }
                }
                retval = false;
            }
        }

    } else {
        // communication failed without a disconnect being received - terminate
        if (this->terminateRequested == false) {
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "MDDriver communication lost - sending disconnect signal and closing connection.");
        }
        retval = false;
    }
    return retval;
}


/*
 * MDDriverConnector::sendForces
 */
bool MDDriverConnector::sendForces(void) {
    using megamol::core::utility::log::Log;

    bool retval = true;
    this->header.type = MDD_MDCOMM;
    this->header.length = this->forcesCount; // fill the header with the correct type and the number of forces

    retval &= this->sendHeader(); // byteswap the header and send it

    this->forceIDs.Lock(); // lock the data arrays
    this->forceList.Lock();

    // send the atom indices data first
    try {
        this->socket.Send(this->forceIDs.PeekElements(), this->forcesCount * sizeof(unsigned int), TIMEOUT, 0, true);
    } catch (vislib::net::SocketException e) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Socket Exception during forces (atom indices) send: %s", e.GetMsgA());
        retval = false;
    }

    // send the atom forces data next
    try {
        this->socket.Send(this->forceList.PeekElements(), this->forcesCount * 3 * sizeof(float), TIMEOUT, 0, true);
    } catch (vislib::net::SocketException e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Socket Exception during forces (force list) send: %s", e.GetMsgA());
        retval = false;
    }

    this->forceIDs.Unlock();
    this->forceList.Unlock(); // unlock the data arrays
    return retval;
}


/*
 * MDDriverConnector::sendPause
 */
bool MDDriverConnector::sendPause(void) {
    this->header.type = MDD_PAUSE;
    this->header.length = 0;   // fill the header with the correct data
    return this->sendHeader(); // byteswap the header and send it
}


/*
 * MDDriverConnector::sendGo
 */
bool MDDriverConnector::sendGo(void) {
    bool retval = true;
    this->header.type = MDD_GO;
    this->header.length = 0;     // fill the header with the correct data
    retval = this->sendHeader(); // byteswap the header and send it

    return retval;
}


/*
 * MDDriverConnector::sendTransferRate
 */
bool MDDriverConnector::sendTransferRate(void) {
    this->header.type = MDD_TRATE;
    this->header.length = this->rateRequested; // fill the header with the rate and header type
    return this->sendHeader();                 // byteswap the header and send it
}


/*
 * MDDriverConnector::getHeader
 */
bool MDDriverConnector::getHeader(void) {
    using megamol::core::utility::log::Log;
    try {
        int errorlevel;
        errorlevel = static_cast<int>(this->socket.Receive(&this->header, sizeof(MDDHeader), TIMEOUT, 0, true));
        if (errorlevel != sizeof(MDDHeader)) {
            if (errorlevel == 0) {
                // no data was received at all - the simulation probably failed catastrophically without warning
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "getHeader: received no header - MDDriver or the simulation "
                                                           "may have failed catastrophically without warning.");
                return false;
            } else {
                // partial header was received -- one fix here might be to loop the socket receive instruction until it receives a full header
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "getHeader: received only partial header.");
                return false;
            }
        }
        this->header.type = this->byteSwap(
            this->header.type); // for reasons unknown, MDDriver automatically byteswaps all the headers it sends out
        this->header.length = this->byteSwap(this->header.length);
    } catch (vislib::net::SocketException e) {
        if (this->reset >= RESET_ATTEMPTS) {
            // if MDDriver has already reached its reset attempts this thread, go ahead and print error messages (otherwise attempt a reset)
            if (this->terminateRequested == false) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Socket Exception during header receive: %s", e.GetMsgA());
                return false;
            }
        }
    }
    return true;
}

/*
 * MDDriverConnector::sendHeader
 */
bool MDDriverConnector::sendHeader(void) {
    using megamol::core::utility::log::Log;
    try {
        this->header.type = this->byteSwap(
            this->header.type); // for reasons unknown, MDDriver wants all headers byteswapped before being sent to it
        this->header.length = this->byteSwap(this->header.length);
        int errorlevel;
        errorlevel = static_cast<int>(this->socket.Send(&this->header, sizeof(MDDHeader), TIMEOUT, 0, true));
        if (errorlevel != sizeof(MDDHeader)) {
            if (errorlevel == 0) {
                // no data was able to be sent at all - possible that socket was not set up correctly
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "sendHeader: unable to send any header data.");
                return false;
            } else {
                // partial header was sent - might need to loop the socket send instruction until full header is sent
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "sendHeader: sent only partial header.");
                return false;
            }
        }
    } catch (vislib::net::SocketException e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Socket Exception during send: %s", e.GetMsgA());
        return false;
    }
    return true; // default
}

/*
 * MDDriverConnector::byteSwap
 */
int MDDriverConnector::byteSwap(int input) {
    char output[4];
    output[0] = ((char*)&input)[3];
    output[1] = ((char*)&input)[2];
    output[2] = ((char*)&input)[1];
    output[3] = ((char*)&input)[0];
    return *((int*)output);
}
