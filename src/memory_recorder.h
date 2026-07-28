#pragma once
#include <atomic>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>

static long getVmRSSKB() {
    std::ifstream file("/proc/self/status");
    std::string line;

    while (std::getline(file, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long value = 0;
            sscanf(line.c_str(), "%*s %ld", &value);
            return value;
        }
    }

    return -1;
}

class MemoryRecorder {
public:
    MemoryRecorder(const std::string& filename, int intervalMs = 50)
        : running_(true), out_(filename) {
        out_ << "time_ms,vmrss_kb,vmrss_mb\n";
        start_ = std::chrono::steady_clock::now();

        worker_ = std::thread([this, intervalMs]() {
            while (running_) {
                auto now = std::chrono::steady_clock::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - start_
                ).count();

                long rssKB = getVmRSSKB();
                out_ << ms << "," << rssKB << "," << rssKB / 1024.0 << "\n";

                std::this_thread::sleep_for(
                    std::chrono::milliseconds(intervalMs)
                );
            }
        });
    }

    ~MemoryRecorder() {
        running_ = false;
        if (worker_.joinable()) {
            worker_.join();
        }
        out_.flush();
    }

private:
    std::atomic<bool> running_;
    std::ofstream out_;
    std::thread worker_;
    std::chrono::steady_clock::time_point start_;
};