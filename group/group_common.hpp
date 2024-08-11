#pragma once
#include "common.h"
#include "group_common.hpp"
#include <iostream>

#include <type_traits>


namespace s = cl::sycl;

template<size_t S>
struct Big {
  constexpr static bool isBig = true;
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t l;
  std::array<uint64_t, S> a;

  bool operator==(const Big& other) const noexcept {
    return other.x == x and other.y == y and other.z == z and other.l == l;
  }

  Big& operator=(const Big& other) noexcept {
    x = other.x;
    y = other.y;
    z = other.z;
    l = other.l;
    a = other.a;
    return *this;
  }

  Big& operator=(uint64_t x) noexcept {
    this->x = x;
    return *this;
  }

  Big& operator+=(const Big& x) noexcept {
    this->x += x.x;
    this->y += y;
    this->z += z;
    this->a += a;
    return *this;
  }

  Big operator+(const Big& other) const noexcept {
    return {x + other.x, y + other.y, z + other.z, l + other.l};
  }
};

template<typename T>
using array_entry_t = typename std::remove_reference<decltype( std::declval<T>()[0] )>::type;

template <typename T>
using elementType = std::remove_reference_t<decltype(T{}.s0())>;

template <typename T, int N>
std::string type_to_string(s::vec<T, N> v) {
  std::stringstream ss{};

  ss << "(";
  if constexpr(1 <= N)
    ss << +v.s0();
  if constexpr(2 <= N)
    ss << ", " << +v.s1();
  if constexpr(3 <= N)
    ss << ", " << +v.s2();
  if constexpr(4 <= N)
    ss << ", " << +v.s3();
  if constexpr(8 <= N) {
    ss << ", " << +v.s4();
    ss << ", " << +v.s5();
    ss << ", " << +v.s6();
    ss << ", " << +v.s7();
  }
  if constexpr(16 <= N) {
    ss << ", " << +v.s8();
    ss << ", " << +v.s9();
    ss << ", " << +v.sA();
    ss << ", " << +v.sB();
    ss << ", " << +v.sC();
    ss << ", " << +v.sD();
    ss << ", " << +v.sE();
    ss << ", " << +v.sF();
  }
  ss << ")";

  return ss.str();
}

template <typename T>
std::string type_to_string(T x) {
  std::stringstream ss{};
  ss << +x;

  return ss.str();
}

template<size_t S>
std::string type_to_string(Big<S> x) {
  std::stringstream ss{};
  ss << x.x << " " << x.y << " " << x.z << " " << x.l;

  return ss.str();
}

template <typename T, typename std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET T initialize_type(T init) {
  return init;
}

template <typename T, typename std::enable_if_t<T::isBig, int> = 0>
HIPSYCL_KERNEL_TARGET T initialize_type(uint32_t init) {
  return {init, init+1, init+2, init+3};
}


template <typename T>
bool compare_type(T x1, T x2) {
  return x1 == x2;
}
