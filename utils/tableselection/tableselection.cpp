#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <zmq.hpp>

class TableSelection {
public:
    TableSelection() {
        context_ = std::make_unique<zmq::context_t>(1);

        zeromq_rx_quit_ = false;
        if (!zeromq_rx_thread_.joinable()) {
            zeromq_rx_thread_ = std::thread(&TableSelection::zeromqRx, this);
        }

        zeromq_tx_quit_ = false;
        zeromq_tx_notified_ = false;
        if (!zeromq_tx_thread_.joinable()) {
            zeromq_tx_thread_ = std::thread(&TableSelection::zeromqTx, this);
        }
    }

    ~TableSelection() {
        zeromq_rx_quit_ = true;
        zeromq_tx_quit_ = true;
        zeromq_tx_notified_ = true;
        zeromq_tx_cond_var_.notify_one();
        context_->close();
        context_.reset();
        if (zeromq_rx_thread_.joinable()) {
            zeromq_rx_thread_.join();
        }
        if (zeromq_tx_thread_.joinable()) {
            zeromq_tx_thread_.join();
        }
    }

    void send(std::vector<uint64_t> ids) {
        std::unique_lock<std::mutex> lock(zeromq_tx_mutex_);
        data_ = std::move(ids);
        zeromq_tx_notified_ = true;
        zeromq_tx_cond_var_.notify_one();
        lock.unlock();
    }

protected:
    void zeromqRx() {
        zmq::socket_t socket{*context_, ZMQ_REP};
        socket.bind("tcp://*:10001");

        const std::string okString{"Ok!"};

        while (!zeromq_rx_quit_) {
            try {
                zmq::message_t request;
                socket.recv(request, zmq::recv_flags::none);
                size_t size = request.size() / sizeof(uint64_t);
                uint64_t* data_ptr = static_cast<uint64_t*>(request.data());
                std::vector<uint64_t> data(data_ptr, data_ptr + size);

                std::cout << "Received ids: ";
                for (uint64_t id : data) {
                    std::cout << id << " ";
                }
                std::cout << std::endl;

                zmq::message_t reply{okString.cbegin(), okString.cend()};
                socket.send(reply, zmq::send_flags::none);
            } catch (const zmq::error_t& e) {
                if (e.num() != ETERM) {
                    std::cerr << e.what() << std::endl;
                }
            }
        }

        std::cout << "zeromqRx() done!" << std::endl;
    }

    void zeromqTx() {
        zmq::socket_t socket{*context_, ZMQ_REQ};
        socket.setsockopt(ZMQ_LINGER, 0);
        socket.connect("tcp://localhost:10002");

        std::unique_lock<std::mutex> lock(zeromq_tx_mutex_);
        while (!zeromq_tx_quit_) {
            while (!zeromq_tx_notified_ && !zeromq_tx_quit_) {
                zeromq_tx_cond_var_.wait(lock);
            }
            while (zeromq_tx_notified_ && !zeromq_tx_quit_) {
                zeromq_tx_notified_ = false;
                try {
                    zmq::message_t request{data_.cbegin(), data_.cend()};
                    lock.unlock();
                    socket.send(request, zmq::send_flags::none);

                    zmq::message_t reply{};
                    socket.recv(reply, zmq::recv_flags::none);
                } catch (const zmq::error_t& e) {
                    if (e.num() != ETERM) {
                        std::cerr << e.what() << std::endl;
                    }
                }
                lock.lock();
            }
        }

        std::cout << "zeromqTx() done!" << std::endl;
    }

    // ZeroMQ
    std::unique_ptr<zmq::context_t> context_;
    std::thread zeromq_rx_thread_;
    bool zeromq_rx_quit_;
    std::thread zeromq_tx_thread_;
    bool zeromq_tx_quit_;
    std::mutex zeromq_tx_mutex_;
    std::condition_variable zeromq_tx_cond_var_;
    bool zeromq_tx_notified_;
    std::vector<uint64_t> data_;
};


int main(int argc, char* argv[]) {
    std::cout << "Table Selection Example" << std::endl;

    TableSelection s;

    std::this_thread::sleep_for(std::chrono::milliseconds(10000));

    std::cout << "Send selection" << std::endl;
    s.send({1, 3, 5});

    std::this_thread::sleep_for(std::chrono::milliseconds(10000));

    return 0;
}
