/*
 * MDDriverConnector.h
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_MDDRIVERCONNECTOR_H_INCLUDED
#define MMPROTEINPLUGIN_MDDRIVERCONNECTOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/param/ParamSlot.h"
#include "vislib/Array.h"
#include "vislib/RawStorage.h"
#include "vislib/net/Socket.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/Runnable.h"

#define MDD_VERSION 2 // this value corresponds to MDDriver ver 1.2 2008-06-25

namespace megamol {
namespace protein {

/**
 * Data source for atom position arrays from real-time simulations.
 * Obtains data via TCP connection to MDDriver.
 *
 * TODO: Suggested code from MDDriver has client-side checks for endianness.
 * Current implementation assumes the same endian on both machines. Future versions could check for endianness.
 */

class MDDriverConnector : public vislib::sys::Runnable {
public:
    /*
     * Table of energies sent from MDDriver.
     * This table was taken directly from MDDriver code ("imd.h" version 1.2 2008-06-25)
     */
    struct MDDEnergies {
        int tstep;    //!< integer timestep index
        float T;      //!< Temperature in degrees Kelvin
        float Etot;   //!< Total energy, in Kcal/mol
        float Epot;   //!< Potential energy, in Kcal/mol
        float Evdw;   //!< Van der Waals energy, in Kcal/mol
        float Eelec;  //!< Electrostatic energy, in Kcal/mol
        float Ebond;  //!< Bond energy, Kcal/mol
        float Eangle; //!< Angle energy, Kcal/mol
        float Edihe;  //!< Dihedral energy, Kcal/mol
        float Eimpr;  //!< Improper energy, Kcal/mol
    };


    /*
     * The 8 byte header that is transferred to and from MDDriver with each new set of data.
     */
    struct MDDHeader {
        unsigned int
            type; // this is always associated with a type from the MDDHeaderType enum, but needs to be an int for byte swapping and to make it easier to send over networks.
        unsigned int
            length; // the length (in number of atoms) of the coordinates list being transferred, or other special values
    };

    /** Ctor */
    MDDriverConnector(void);

    /** Dtor */
    ~MDDriverConnector(void) override;

    /**
     * Startup callback of the thread. The Thread class will call that
     * before Run().
     *
     * @param config A pointer to the Configuration, which specifies the
     *               settings of the connector.
     */
    void OnThreadStarting(void* config) override;

    /**
     * Perform the work of a thread.
     *
     * @param config A pointer to the Configuration, which specifies the
     *               settings of the connector.
     *
     * @return The application dependent return code of the thread. This
     *         must not be STILL_ACTIVE (259).
     */
    DWORD Run(void* config) override;

    /**
     * Abort the work of the connector by forcefully closing the underlying
     * communication channel.
     *
     * @return true.
     */
    bool Terminate(void) override;


    /**
     * Checks whether or not the socket is functional.
     *
     * @return 'true' if valid, 'false' otherwise.
     */
    bool IsSocketFunctional(void) {
        return this->socketValidity;
    }

    /**
     * Sets the host and the port for the socket connection.
     * This MUST be called before calling Start for the thread, or the socket
     * will never get set up correctly.
     *
     * @param inHost Host name or IP.
     * @param inPort Port number.
     */
    void Initialize(vislib::TString inHost, int inPort);

    /**
     * Requests that the thread pause MDDriver. Does nothing if MDDriver is already paused.
     * No new data will be acquired while the simulation is paused, but the old data
     * is saved. Also, transfer rate, forces, and termination commands can be send
     * while the simulation is paused.
     */
    inline void RequestPause(void) {
        this->pauseRequested = true;
    }

    /**
     * Requests that the thread start up MDDriver from a paused state.
     * Does nothing if MDDriver is already running.
     */
    inline void RequestGo(void) {
        this->goRequested = true;
    }

    /**
     * Requests the thread update the MDDriver transfer rate to the value of the input
     * parameter. Transfer rate is in ms.
     *
     * @param inRate New time between transfers (in ms)
     */
    inline void RequestTransferRate(int inRate) {
        this->rateRequested = inRate;
    }

    /**
     * Requests that the thread disconnect the socket connection and stop running.
     * Use IsRunning() to check if the thread has terminated. A new thread may be
     * started after terminating the old thread.
     */
    inline void RequestTerminate(void) {
        this->terminateRequested = true;
    }

    /**
     * Requests that the thread send the forces specified in the parameters to
     * MDDriver to be used in the simulation.
     *
     * @param count The number of forces to be sent (i.e. the number of atoms on
     * which forces are being applied).
     * @param atomIDs Pointer to an array of ints giving the atom ids that correspond
     * to the forces in the forces array.
     * @param forces Pointer to an array of x,y,z,x,y,z floats that give the forces
     * for the atom ids in the atomIDs array.
     */
    void RequestForces(int count, const unsigned int* atomIDs, const float* forces);

    /**
     * Gets the most recent atom position data and fills it into the array
     * pointed to by atomPos. The array should expect count * 3 floats of coordinates
     * in x,y,z,x,y,z form. If the array size expected does not match the array size
     * received from MDDriver, this method prints an error and does not copy data.
     *
     * @param count The number of atoms for which coordinate data is expected.
     * @param atomPos Pointer to array of at least 3*count float space into which
     * atom coordinate data will be copied.
     */
    void GetCoordinates(int count, float* atomPos);

    /**
     * Releases socket resources.
     */
    void release(void);

private:
    /*
     * This list was taken from the MDDriver code ("imd.h" version 1.2 2008-06-25)
     * The list corresponds to the header value transferred by MDDriver, and specifies
     * what kind of data is to be expected in the following data transfer. It is also
     * used to send commands to MDDriver.
     */
    enum MDDHeaderType {
        MDD_DISCONNECT, //!< close IMD connection, leaving sim running
        MDD_ENERGIES,   //!< energy data block
        MDD_COORDS,     //!< atom coordinates
        MDD_GO,         //!< start the simulation
        MDD_HANDSHAKE,  //!< endianism and version check message
        MDD_KILL,       //!< kill the simulation job, shutdown IMD
        MDD_MDCOMM,     //!< MDComm style force data
        MDD_PAUSE,      //!< pause the running simulation
        MDD_TRATE,      //!< set IMD update transmission rate
        MDD_IOERROR     //!< indicate an I/O error
    };

    /**
     * Starts the socket connection with the given host and port; performs handshaking.
     * Sets the socket validity flag to true if it succeeds.
     *
     * @param host String representing either name or IP of the machine running MDDriver.
     * @param port Port number to communicate with on the server machine.
     * @return 'true' on success, 'false' otherwise.
     */
    bool startSocket(const vislib::TString& host, int port);

    /**
     * Gets atom coordinates and energy data from MDDriver
     *
     * @return True on success.
     */
    bool getData(void);


    /**
     * Sends atom indices list and forces list to MDDriver. Note that not all atoms must be send - the atom numbers
     * in the atomIndices array will correspond to the forces in the forceList array.
     *
     * @param numForces Number of forces being sent by caller.
     *
     * @return True on success.
     */
    bool sendForces(void);


    /**
     * Sends a pause signal to the simulator.
     *
     * @return True on success.
     */
    bool sendPause(void);


    /**
     * Sends a go (unpause) signal to the simulator.
     *
     * @return True on success.
     */
    bool sendGo(void);


    /**
     * Sends the desired data transfer rate to the simulator.
     *
     * @param rate The time between transfers in ms
     *
     * @return True on success.
     */
    bool sendTransferRate(void);


    /**
     * Receives header data from MDDriver and byteswaps it
     *
     * @return True on success.
     */
    bool getHeader(void);

    /**
     * Sends header data to MDDriver after byteswapping it.
     *
     * @return True on success.
     */
    bool sendHeader(void);

    /**
     * Switches order of bytes in an int so it changes endian
     */
    int byteSwap(int input);

    // -------------------- variables --------------------

    /** The socket for MD Driver connection */
    vislib::net::Socket socket;

    /** The socket status */
    bool socketValidity;

    /** The energies table */
    MDDEnergies energies;

    /** The header */
    MDDHeader header;

    /** Flag set if MDDriver is currently paused (prevents get data) */
    bool paused;

    /** Number of atom coordinates received from MDDriver */
    int atomCount;

    /** The atom coordinates */
    vislib::Array<float, vislib::sys::CriticalSection> atomCoordinates;

    /** Number of forces to be sent */
    int forcesCount;

    /** The atom id list for forces */
    vislib::Array<int, vislib::sys::CriticalSection> forceIDs;

    /** The list of actual forces */
    vislib::Array<float, vislib::sys::CriticalSection> forceList;

    /** Flag if the forces have been zeroed out since the last time forces were applied and removed */
    bool zeroForce;

    /** The host name to which the thread socket will connect */
    vislib::TString host;

    /** The port for the socket connection */
    int port;

    /** Flag requesting that the thread pause MDDriver */
    bool pauseRequested;

    /** Flag requesting that the thread start MDDriver */
    bool goRequested;

    /** Flag requesting that the thread terminate the socket and then itself */
    bool terminateRequested;

    /** Flag requesting that the thread change the transfer rate (0 means no change) */
    int rateRequested;

    /** Number of times MDDriver has been reset (paused and unpaused) in this thread */
    int reset;
};

} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_MDDRIVERCONNECTOR_H_INCLUDED
