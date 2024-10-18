#pragma once
#include "common.h"
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

struct Byte16 {
  std::array<uint64_t, 2> a;

  Byte16() = default;
  Byte16(std::array<uint64_t, 2> x) : a(x) {}
  Byte16(const Byte16& other) : a(other.a) {}

  bool operator==(const Byte16& other) const noexcept {
    return other.a == a;
  }

  Byte16& operator=(const Byte16& other) noexcept {
    a = other.a;
    return *this;
  }

  Byte16& operator=(Byte16&& other) noexcept {
    a = other.a;
    return *this;
  }
};

struct Byte32 {
  std::array<uint64_t, 4> a;

  Byte32() = default;
  Byte32(std::array<uint64_t, 4> x) : a(x) {}
  Byte32(const Byte32& other) : a(other.a) {}

  bool operator==(const Byte32& other) const noexcept {
    return other.a == a;
  }

  Byte32& operator=(const Byte32& other) {
    a = other.a;
    return *this;
  }

  Byte32& operator=(Byte32&& other) noexcept {
    a = other.a;
    return *this;
  }
};

struct Byte64 {
  std::array<uint64_t, 8> a;

  Byte64() = default;
  Byte64(std::array<uint64_t, 8> x) : a(x) {}
  Byte64(const Byte64& other) : a(other.a) {}

  bool operator==(const Byte64& other) const noexcept {
    return other.a == a;
  }

  Byte64& operator=(const Byte64& other) {
    a = other.a;
    return *this;
  }

  Byte64& operator=(Byte64&& other) noexcept {
    a = other.a;
    return *this;
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
  ss << x;

  return ss.str();
}

template <>
std::string type_to_string(Byte16 x) {
  std::stringstream ss{};
  ss << x.a[0] << x.a[1];

  return ss.str();
}

template <>
std::string type_to_string(Byte32 x) {
  std::stringstream ss{};
  ss << x.a[0] << x.a[1] << x.a[2] << x.a[3];

  return ss.str();
}

template <>
std::string type_to_string(Byte64 x) {
  std::stringstream ss{};
  ss << x.a[0] << x.a[1] << x.a[2] << x.a[3];

  return ss.str();
}

template<size_t S>
std::string type_to_string(Big<S> x) {
  std::stringstream ss{};
  ss << x.x << " " << x.y;

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


template <typename T, typename std::enable_if_t<std::is_same_v<T, Byte16>, int> = 0>
HIPSYCL_KERNEL_TARGET T initialize_type(uint32_t init) {
  const auto v = static_cast<uint64_t>(init);
  return Byte16{std::array{v, v}};
}

template <typename T, typename std::enable_if_t<std::is_same_v<T, Byte32>, int> = 0>
HIPSYCL_KERNEL_TARGET T initialize_type(uint32_t init) {
  auto v = static_cast<uint64_t>(init);
  return Byte32{std::array{v, v+1, v+2, v+3}};
}

template <typename T, typename std::enable_if_t<std::is_same_v<T, Byte64>, int> = 0>
HIPSYCL_KERNEL_TARGET T initialize_type(uint32_t init) {
  auto v = static_cast<uint64_t>(init);
  return Byte64{std::array{v, v+1, v+2, v+3, v+4, v+5, v+6, v+7}};
}

template <typename T>
bool compare_type(T x1, T x2) {
  return x1 == x2;
}