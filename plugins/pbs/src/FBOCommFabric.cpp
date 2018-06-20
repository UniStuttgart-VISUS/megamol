#include "stdafx.h"
#include "FBOCommFabric.h"

#include <mpi.h>


megamol::pbs::MPICommFabric::MPICommFabric(int target_rank, int source_rank)
    : my_rank_{0}
    , target_rank_{target_rank}
    , source_rank_{source_rank}
    , recv_count_{1} {
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);
}


bool megamol::pbs::MPICommFabric::Connect(std::string const &address) { return true; }


bool megamol::pbs::MPICommFabric::Bind(std::string const &address) { return true; }


bool megamol::pbs::MPICommFabric::Send(std::vector<char> const& buf, send_type const type) {
    auto status = MPI_Send(buf.data(), buf.size(), MPI_CHAR, target_rank_, 0, MPI_COMM_WORLD);
    return status == MPI_SUCCESS;
}


bool megamol::pbs::MPICommFabric::Recv(std::vector<char> &buf, recv_type const type) {
    MPI_Status stat;
    buf.resize(recv_count_);
    auto status = MPI_Recv(buf.data(), recv_count_, MPI_CHAR, source_rank_, 0, MPI_COMM_WORLD, &stat);
    MPI_Get_count(&stat, MPI_CHAR, &recv_count_);
    return status == MPI_SUCCESS;
}


bool megamol::pbs::MPICommFabric::Disconnect() { return true; }


megamol::pbs::MPICommFabric::~MPICommFabric() {  }


megamol::pbs::ZMQCommFabric::ZMQCommFabric(zmq::socket_type const& type) : ctx_{1}, socket_{ctx_, type} {}


megamol::pbs::ZMQCommFabric::ZMQCommFabric(ZMQCommFabric&& rhs) noexcept
    : ctx_{std::move(rhs.ctx_)}, socket_{std::move(rhs.socket_)} {}


megamol::pbs::ZMQCommFabric& megamol::pbs::ZMQCommFabric::operator=(ZMQCommFabric&& rhs) noexcept {
    this->ctx_ = std::move(rhs.ctx_);
    this->socket_ = std::move(rhs.socket_);
    return *this;
}


bool megamol::pbs::ZMQCommFabric::Connect(std::string const& address) {
    /*if (!this->socket_.connected()) {
        this->address_ = address;
        this->socket_.connect(address);
        return this->socket_.connected();
    }
    return true;*/
    this->address_ = address;
    this->socket_.connect(address);
    return this->socket_.connected();
}


bool megamol::pbs::ZMQCommFabric::Bind(std::string const& address) {
    this->address_ = address;
    try {
        this->socket_.bind(address);
    } catch (zmq::error_t const& e) {
        printf("ZMQ ERROR: %s", e.what());
    }
    return this->socket_.connected();
}


bool megamol::pbs::ZMQCommFabric::Send(std::vector<char> const& buf, send_type const type) {
    return this->socket_.send(buf.begin(), buf.end());
}


bool megamol::pbs::ZMQCommFabric::Recv(std::vector<char>& buf, recv_type const type) {
    zmq::message_t msg;
    auto const ret = this->socket_.recv(&msg);
    if (!ret) return false;
    buf.resize(msg.size());
    std::copy(static_cast<char*>(msg.data()), static_cast<char*>(msg.data()) + msg.size(), buf.begin());
    return true;
}


bool megamol::pbs::ZMQCommFabric::Disconnect() {
    if (this->socket_.connected()) {
        this->socket_.disconnect(this->address_);
        return !this->socket_.connected();
    }
    return true;
}


megamol::pbs::ZMQCommFabric::~ZMQCommFabric() { this->Disconnect(); }


megamol::pbs::FBOCommFabric::FBOCommFabric(std::unique_ptr<AbstractCommFabric>&& pimpl)
    : pimpl_{std::forward<std::unique_ptr<AbstractCommFabric>>(pimpl)} {}


bool megamol::pbs::FBOCommFabric::Connect(std::string const& address) { return this->pimpl_->Connect(address); }


bool megamol::pbs::FBOCommFabric::Bind(std::string const& address) { return this->pimpl_->Bind(address); }


bool megamol::pbs::FBOCommFabric::Send(std::vector<char> const& buf, send_type const type) {
    return this->pimpl_->Send(buf, type);
}


bool megamol::pbs::FBOCommFabric::Recv(std::vector<char>& buf, recv_type const type) {
    return this->pimpl_->Recv(buf, type);
}


bool megamol::pbs::FBOCommFabric::Disconnect() { return this->pimpl_->Disconnect(); }
