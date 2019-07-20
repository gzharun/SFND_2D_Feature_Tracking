/**
 * Copyright (C) 2017-2018, Ring.
 *
 * All rights reserved. No Part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Ring.
 */

#pragma once

#include <vector>

/// This circular buffer data structure
template<typename T, size_t N>
class CircularBuffer {
  public:
    /// constructor
    CircularBuffer();
    ~CircularBuffer() = default;

    ////////////////////////////////////////////////////
    /// ... state functions ...
    ////////////////////////////////////////////////////
    /// checks if queue is empty
    bool empty() const;
    /// checks if queue is full
    bool full() const;
    /// returns current capacity of the queue
    constexpr int capacity() const;
    /// returns current amount of elements in the queue
    int size() const;

    ////////////////////////////////////////////////////////////////////
    /// gets last added element
    T& back();
    /// push an element to the queue
    ///
    /// \param el new element
    /// \return true on success
    bool push(const T& el);
    /// push an element to the queue (optimised for movable objects)
    ///
    /// \param el new element
    /// \return true on success
    bool push(T&& el);

    ////////////////////////////////////////////////////////////////////
    /// gets the first element
    T& front();
    /// remove the first element form the queue
    void pop();
    /// gets the first element extracting it from the queue (optimised for movable objects)
    ///
    /// \return extracted element if the queue is not empty, behavior is undefined otherwise
    T extract();
    /// clear the queue
    void clear();

  private:
    void advance();
    void retreat();

    bool full_;

    /// Index for next write
    int write_idx_;
    /// Index for next read
    int read_idx_;

    std::vector<T> buffer_;
};

template<typename T, size_t N>
inline CircularBuffer<T, N>::CircularBuffer()
    : full_(false)
    , write_idx_(0)
    , read_idx_(0)
    , buffer_(std::vector<T>(N)) {
    }

template<typename T, size_t N>
inline bool CircularBuffer<T, N>::empty() const {
    return (write_idx_ == read_idx_) && !full_;
}

template<typename T, size_t N>
inline bool CircularBuffer<T, N>::full() const {
    return full_ || !capacity();
}

template<typename T, size_t N>
constexpr inline int CircularBuffer<T, N>::capacity() const {
    return buffer_.size();
}

template<typename T, size_t N>
inline int CircularBuffer<T, N>::size() const {
    if (!full()) {
        int max_size = capacity();
        return (max_size + write_idx_ - read_idx_) % max_size;
    }

    return capacity();
}

template<typename T, size_t N>
inline void CircularBuffer<T, N>::advance() {
    write_idx_ = (write_idx_ + 1) % capacity();
    full_ = (write_idx_ == read_idx_);
}

template<typename T, size_t N>
inline void CircularBuffer<T, N>::retreat() {
    full_ = false;
    read_idx_ = (read_idx_ + 1) % capacity();
}

template<typename T, size_t N>
inline T& CircularBuffer<T, N>::back() {
    size_t back_idx = (write_idx_ - 1 + capacity()) % capacity();
    return buffer_[back_idx];
}

template<typename T, size_t N>
inline T& CircularBuffer<T, N>::front() {
    return buffer_[read_idx_];
}

template<typename T, size_t N>
inline bool CircularBuffer<T, N>::push(const T& el) {
    if (full())
        pop();

    buffer_[write_idx_] = el;
    advance();

    return true;
}

template<typename T, size_t N>
inline bool CircularBuffer<T, N>::push(T&& el) {
    if (full())
        pop();

    buffer_[write_idx_] = std::move(el);
    advance();

    return true;
}

template<typename T, size_t N>
inline void CircularBuffer<T, N>::pop() {
    if (empty())
        return;

    retreat();
}

template<typename T, size_t N>
inline T CircularBuffer<T, N>::extract() {
    auto ret = std::move(buffer_[read_idx_]);
    retreat();
    return ret;
}

template<typename T, size_t N>
inline void CircularBuffer<T, N>::clear() {
    full_ = false;
    read_idx_ = write_idx_;
}
