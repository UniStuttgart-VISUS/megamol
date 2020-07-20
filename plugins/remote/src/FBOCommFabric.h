#pragma once

#include <memory>
#include <vector>
#ifdef WITH_MPI
#include <mpi.h>
#endif
#include "zmq.hpp"

namespace megamol {
namespace remote {

enum send_type : unsigned int { ST_UNDEF = 0, BCAST, SCATTER, SEND, ISEND };

enum recv_type : unsigned int { RT_UNDEF = 0, RECV, IRECV };


class AbstractCommFabric {
public:
    virtual bool Connect(std::string const& address) = 0;
    virtual bool Bind(std::string const& address) = 0;
    virtual bool Send(std::vector<char> const& buf, send_type const type = ST_UNDEF) = 0;
    virtual bool Recv(std::vector<char>& buf, recv_type const type = RT_UNDEF) = 0;
    virtual bool Disconnect(void) = 0;
    virtual ~AbstractCommFabric(void) = default;
};


/**
 * Comm layer for MPI
 */
class MPICommFabric : public AbstractCommFabric {
public:
    MPICommFabric(int target_rank, int source_rank);
    bool Connect(std::string const& address) override;
    bool Bind(std::string const& address) override;
    bool Send(std::vector<char> const& buf, send_type const type = ST_UNDEF) override;
    bool Recv(std::vector<char>& buf, recv_type const type = RT_UNDEF) override;
    bool Disconnect() override;
    virtual ~MPICommFabric(void);
private:
    int my_rank_;

    int target_rank_;

    int source_rank_;

    int recv_count_;
};

/**
 * Comm layer for ZeroMQ
 */
class ZMQCommFabric : public AbstractCommFabric {
public:
    ZMQCommFabric(zmq::socket_type const& type);
    ZMQCommFabric(ZMQCommFabric const& rhs) = delete;
    ZMQCommFabric& operator=(ZMQCommFabric const& rhs) = delete;
    ZMQCommFabric(ZMQCommFabric&& rhs) noexcept;
    ZMQCommFabric& operator=(ZMQCommFabric&& rhs) noexcept;
    bool Connect(std::string const& address) override;
    bool Bind(std::string const& address) override;
    bool Send(std::vector<char> const& buf, send_type const type = ST_UNDEF) override;
    bool Recv(std::vector<char>& buf, recv_type const type = RT_UNDEF) override;
    bool Disconnect(void) override;
    virtual ~ZMQCommFabric(void);

private:
    zmq::context_t ctx_;
    zmq::socket_t socket_;
    /** endpoint address to which the socket is connected to */
    std::string address_;
    bool bound_ = false;
};


/**
 * Comm layer for WebSocket
 */
class WSCommFabric : public AbstractCommFabric {};


class FBOCommFabric : public AbstractCommFabric {
public:
    enum commtype { ZMQ_COMM, MPI_COMM };

    FBOCommFabric(std::unique_ptr<AbstractCommFabric>&& pimpl);

    FBOCommFabric(FBOCommFabric const& rhs) = delete;

    FBOCommFabric& operator=(FBOCommFabric const& rhs) = delete;

    FBOCommFabric(FBOCommFabric&& rhs) noexcept = default;

    FBOCommFabric& operator=(FBOCommFabric&& rhs) noexcept = default;

    bool Connect(std::string const& address) override;

    bool Bind(std::string const& address) override;

    bool Send(std::vector<char> const& buf, send_type const type = ST_UNDEF) override;

    bool Recv(std::vector<char>& buf, recv_type const type = RT_UNDEF) override;

    bool Disconnect(void) override;

    virtual ~FBOCommFabric(void) = default;

protected:
private:
    std::unique_ptr<AbstractCommFabric> pimpl_;
};

} // end namespace remote
} // end namespace megamol
