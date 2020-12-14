#ifndef TYPE_TRAITS_H
#define TYPE_TRAITS_H

#include <CL/sycl.hpp>

template<class T>
struct ReadableTypename
{};

#define MAKE_READABLE_TYPENAME(T, str) \
template<> \
struct ReadableTypename<T> \
{ static const char* name; }; const char* ReadableTypename<T>::name = str;

MAKE_READABLE_TYPENAME(char, "int8")
MAKE_READABLE_TYPENAME(unsigned char, "uint8")
MAKE_READABLE_TYPENAME(short, "int16")
MAKE_READABLE_TYPENAME(unsigned short, "uint16")
MAKE_READABLE_TYPENAME(int, "int32")
MAKE_READABLE_TYPENAME(unsigned int, "uint32")
MAKE_READABLE_TYPENAME(long long, "int64")
MAKE_READABLE_TYPENAME(unsigned long long, "uint64")
MAKE_READABLE_TYPENAME(float, "fp32")
MAKE_READABLE_TYPENAME(double, "fp64")

template<> struct ReadableTypename<cl::sycl::vec<char, 1>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<char,  1>>::name = "vec<int8:1>";
template<> struct ReadableTypename<cl::sycl::vec<char, 2>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<char,  2>>::name = "vec<int8:2>";
template<> struct ReadableTypename<cl::sycl::vec<char, 3>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<char,  3>>::name = "vec<int8:3>";
template<> struct ReadableTypename<cl::sycl::vec<char, 4>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<char,  4>>::name = "vec<int8:4>";
template<> struct ReadableTypename<cl::sycl::vec<char, 8>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<char,  8>>::name = "vec<int8:8>";
template<> struct ReadableTypename<cl::sycl::vec<char, 16>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<char, 16>>::name = "vec<int8:16>";

template<> struct ReadableTypename<cl::sycl::vec<int, 1>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<int,  1>>::name = "vec<int32:1>";
template<> struct ReadableTypename<cl::sycl::vec<int, 2>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<int,  2>>::name = "vec<int32:2>";
template<> struct ReadableTypename<cl::sycl::vec<int, 3>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<int,  3>>::name = "vec<int32:3>";
template<> struct ReadableTypename<cl::sycl::vec<int, 4>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<int,  4>>::name = "vec<int32:4>";
template<> struct ReadableTypename<cl::sycl::vec<int, 8>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<int,  8>>::name = "vec<int32:8>";
template<> struct ReadableTypename<cl::sycl::vec<int, 16>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<int, 16>>::name = "vec<int32:16>";

template<> struct ReadableTypename<cl::sycl::vec<long long, 1>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<long long,  1>>::name = "vec<int64:1>";
template<> struct ReadableTypename<cl::sycl::vec<long long, 2>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<long long,  2>>::name = "vec<int64:2>";
template<> struct ReadableTypename<cl::sycl::vec<long long, 3>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<long long,  3>>::name = "vec<int64:3>";
template<> struct ReadableTypename<cl::sycl::vec<long long, 4>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<long long,  4>>::name = "vec<int64:4>";
template<> struct ReadableTypename<cl::sycl::vec<long long, 8>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<long long,  8>>::name = "vec<int64:8>";
template<> struct ReadableTypename<cl::sycl::vec<long long, 16>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<long long, 16>>::name = "vec<int64:16>";

template<> struct ReadableTypename<cl::sycl::vec<float, 1>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<float,  1>>::name = "vec<fp32:1>";
template<> struct ReadableTypename<cl::sycl::vec<float, 2>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<float,  2>>::name = "vec<fp32:2>";
template<> struct ReadableTypename<cl::sycl::vec<float, 3>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<float,  3>>::name = "vec<fp32:3>";
template<> struct ReadableTypename<cl::sycl::vec<float, 4>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<float,  4>>::name = "vec<fp32:4>";
template<> struct ReadableTypename<cl::sycl::vec<float, 8>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<float,  8>>::name = "vec<fp32:8>";
template<> struct ReadableTypename<cl::sycl::vec<float, 16>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<float, 16>>::name = "vec<fp32:16>";

template<> struct ReadableTypename<cl::sycl::vec<double, 1>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<double,  1>>::name = "vec<fp64:1>";
template<> struct ReadableTypename<cl::sycl::vec<double, 2>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<double,  2>>::name = "vec<fp64:2>";
template<> struct ReadableTypename<cl::sycl::vec<double, 3>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<double,  3>>::name = "vec<fp64:3>";
template<> struct ReadableTypename<cl::sycl::vec<double, 4>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<double,  4>>::name = "vec<fp64:4>";
template<> struct ReadableTypename<cl::sycl::vec<double, 8>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<double,  8>>::name = "vec<fp64:8>";
template<> struct ReadableTypename<cl::sycl::vec<double, 16>> { static const char* name; };
const char* ReadableTypename<cl::sycl::vec<double, 16>>::name = "vec<fp64:16>";

#endif
