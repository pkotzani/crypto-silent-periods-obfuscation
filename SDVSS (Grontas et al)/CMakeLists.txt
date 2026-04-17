cmake_minimum_required(VERSION 3.20)
project(cab_sig_c C)

set(CMAKE_C_STANDARD 11)

find_package(OpenSSL REQUIRED)

add_executable(cab_sig_c main.c)
target_link_libraries(cab_sig_c OpenSSL::Crypto)
