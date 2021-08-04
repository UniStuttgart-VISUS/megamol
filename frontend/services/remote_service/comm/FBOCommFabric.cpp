#include "stdafx.h"
#include "FBOCommFabric.h"

#ifdef WITH_MPI
#include <mpi.h>
#endif // WITH_MPI


megamol::remote::MPICommFabric::MPICommFabric(int target_rank, int source_rank)
    : my_rank_{0}, target_rank_{target_rank}, source_rank_{source_rank}, recv_count_{1} {
    // TODO this is wrong. mpiprovider gives you the correct comm
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);
#endif // WITH_MPI
}


bool megamol::remote::MPICommFabric::Connect(std::string const& address) { return true; }


bool megamol::remote::MPICommFabric::Bind(std::string const& address) { return true; }


bool megamol::remote::MPICommFabric::Send(std::vector<char> const& buf, send_type const type) {
#ifdef WITH_MPI
    // TODO this is wrong. mpiprovider gives you the correct comm
    auto status = MPI_Send((void*)buf.data(), buf.size(), MPI_CHAR, target_rank_, 0, MPI_COMM_WORLD);
    return status == MPI_SUCCESS;
#else
    return false;
#endif // WITH_MPI
}


bool megamol::remote::MPICommFabric::Recv(std::vector<char>& buf, recv_type const type) {
#ifdef WITH_MPI
    MPI_Status stat;
    buf.resize(recv_count_);
    // TODO this is wrong. mpiprovider gives you the correct comm
    auto status = MPI_Recv(buf.data(), recv_count_, MPI_CHAR, source_rank_, 0, MPI_COMM_WORLD, &stat);
    MPI_Get_count(&stat, MPI_CHAR, &recv_count_);
    return status == MPI_SUCCESS;
#else 
    return false;
#endif // WITH_MPI
}


bool megamol::remote::MPICommFabric::Disconnect() { return true; }


megamol::remote::MPICommFabric::~MPICommFabric() {}


megamol::remote::ZMQCommFabric::ZMQCommFabric(zmq::socket_type const& type) : ctx_{1}, socket_{ctx_, type} {}


megamol::remote::ZMQCommFabric::ZMQCommFabric(ZMQCommFabric&& rhs) noexcept
    : ctx_{std::move(rhs.ctx_)}, socket_{std::move(rhs.socket_)} {}


megamol::remote::ZMQCommFabric& megamol::remote::ZMQCommFabric::operator=(ZMQCommFabric&& rhs) noexcept {
    this->ctx_ = std::move(rhs.ctx_);
    this->socket_ = std::move(rhs.socket_);
    return *this;
}


bool megamol::remote::ZMQCommFabric::Connect(std::string const& address) {
    /*if (!this->socket_.connected()) {
        this->address_ = address;
        this->socket_.connect(address);
        return this->socket_.connected();
    }
    return true;*/
    this->address_ = address;
    this->socket_.connect(address);
    // this->socket_.setsockopt(ZMQ_CONFLATE, true);
    this->socket_.setsockopt(ZMQ_LINGER, 0);
    return this->socket_.connected();
}


bool megamol::remote::ZMQCommFabric::Bind(std::string const& address) {
    this->address_ = address;
    try {
        this->socket_.bind(address);
        bound_ = true;
        // this->socket_.setsockopt(ZMQ_CONFLATE, true);
        this->socket_.setsockopt(ZMQ_LINGER, 0);
    } catch (zmq::error_t const& e) {
        printf("ZMQ ERROR: %s", e.what());
    }
    return this->socket_.connected();
}


bool megamol::remote::ZMQCommFabric::Send(std::vector<char> const& buf, send_type const type) {
    return this->socket_.send(buf.begin(), buf.end());
}


bool megamol::remote::ZMQCommFabric::Recv(std::vector<char>& buf, recv_type const type) {
    zmq::message_t msg;
    auto const ret = this->socket_.recv(&msg, ZMQ_DONTWAIT);
    if (!ret) return false;
    buf.resize(msg.size());
    std::copy(static_cast<char*>(msg.data()), static_cast<char*>(msg.data()) + msg.size(), buf.begin());
    return true;
}


bool megamol::remote::ZMQCommFabric::Disconnect() {
    // if (this->socket_.connected()) {
    if (!this->address_.empty()) {
        if (bound_) {
            if (this->address_.find('*') != std::string::npos) {
                // wildcard in the address is not valid for socket unbind
                // get last endpoint manually and unbind from that
                char port[1024];
                size_t size = sizeof(port);
                this->socket_.getsockopt(ZMQ_LAST_ENDPOINT, &port, &size);
                this->address_ = std::string{port, size};
            }
            this->socket_.unbind(this->address_);
        } else {
            this->socket_.disconnect(this->address_);
        }
        return !this->socket_.connected();
    }
    return true;
}


megamol::remote::ZMQCommFabric::~ZMQCommFabric() { /* this->Disconnect(); */ }


megamol::remote::FBOCommFabric::FBOCommFabric(std::unique_ptr<AbstractCommFabric>&& pimpl)
    : pimpl_{std::forward<std::unique_ptr<AbstractCommFabric>>(pimpl)} {}


bool megamol::remote::FBOCommFabric::Connect(std::string const& address) { return this->pimpl_->Connect(address); }


bool megamol::remote::FBOCommFabric::Bind(std::string const& address) { return this->pimpl_->Bind(address); }


bool megamol::remote::FBOCommFabric::Send(std::vector<char> const& buf, send_type const type) {
    return this->pimpl_->Send(buf, type);
}


bool megamol::remote::FBOCommFabric::Recv(std::vector<char>& buf, recv_type const type) {
    return this->pimpl_->Recv(buf, type);
}


bool megamol::remote::FBOCommFabric::Disconnect() { return this->pimpl_->Disconnect(); }
