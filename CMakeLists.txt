cmake_minimum_required(VERSION 3.26)

# NOTE: update executable name in .github/workflows/cmake.yml:25 when changing executable name in this file
# for now, the project name is used as the executable name
set(MAIN_PROJECT_NAME "oop")
set(MAIN_EXECUTABLE_NAME "${MAIN_PROJECT_NAME}")

project(${MAIN_PROJECT_NAME})

# Enable CUDA if available
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CUDA_ENABLED TRUE)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Toolkit: ${CUDAToolkit_LIBRARY_DIR}")
else()
    set(CUDA_ENABLED FALSE)
    message(STATUS "CUDA not found - building without CUDA support")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Collect source files
file(GLOB_RECURSE CPP_HEADERS "${CMAKE_SOURCE_DIR}/include/*.h" "${CMAKE_SOURCE_DIR}/include/*.hpp")

if(CUDA_ENABLED)
    # Only compile .cu files - .cpp files that need CUDA have .cu wrappers
    file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cu")
    file(GLOB_RECURSE CUDA_HEADERS "${CMAKE_SOURCE_DIR}/include/*.cuh")

    set(SOURCES ${CUDA_SOURCES})
    set(HEADERS ${CPP_HEADERS} ${CUDA_HEADERS})

    # Set CUDA architectures for compatibility
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
           CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
            set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "50;60;70;80")
        endif()
    endif()

    # Workaround for MSVC 19.44+ internal compiler error (ICE) with CUDA
    # See: https://developercommunity.visualstudio.com/
    # Error: fatal error C1001 in type_traits / lambdas.cpp
    if(MSVC AND MSVC_VERSION GREATER_EQUAL 1944)
        message(STATUS "Applying MSVC 19.44+ ICE workaround for CUDA")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/d2SSAOptimizer-")
    endif()
else()
    # Non-CUDA build: only pure C++ files
    file(GLOB_RECURSE CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
    set(SOURCES ${CPP_SOURCES})
    set(HEADERS ${CPP_HEADERS})
endif()
file(GLOB_RECURSE TESTS RELATIVE ${CMAKE_SOURCE_DIR} "test/*.cpp")
include_directories(include)
include_directories(src)

option(WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(USE_ASAN "Use Address Sanitizer" OFF)
option(USE_MSAN "Use Memory Sanitizer" OFF)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "install_dir" CACHE PATH "..." FORCE)
endif()

# disable sanitizers when releasing executables without explicitly requested debug info
# use generator expressions to set flags correctly in both single and multi config generators
set(is_debug "$<CONFIG:Debug>")
set(is_rel_with_deb "$<CONFIG:RelWithDebInfo>")
set(debug_mode "$<OR:${is_debug},${is_rel_with_deb}>")

###############################################################################

# custom functions

function(detect_libcpp)
    set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    set(TEST_STDLIB_SRC [=[
    #include <iostream>
    int main() {}
    ]=])
    try_compile(HAS_LIBCPP SOURCE_FROM_CONTENT test_stdlib.cpp ${TEST_STDLIB_SRC})
    set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})
    unset(OLD_CMAKE_CXX_FLAGS)
endfunction()


function(set_custom_stdlib_and_sanitizers target add_apple_asan)
    # Skip sanitizers for CUDA builds - they are incompatible
    if(CUDA_ENABLED)
        message(STATUS "Sanitizers disabled for CUDA build")
        return()
    endif()

    if(MSVC)
        # see https://gitlab.kitware.com/cmake/cmake/-/issues/24922
        set_target_properties(${target} PROPERTIES VS_USER_PROPS "${CMAKE_SOURCE_DIR}/disable_modules.props")
        target_compile_options(${target} PRIVATE /experimental:module-)
        if(USE_ASAN)
            target_compile_options(${target} PRIVATE "$<${debug_mode}:/fsanitize=address>")
        endif()
        return()
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND NOT WIN32)
        detect_libcpp()
        if(HAS_LIBCPP)
            # see also https://stackoverflow.com/a/70382484
            target_compile_options(${target} PRIVATE -stdlib=libc++)
            target_link_options(${target} PRIVATE -stdlib=libc++)
        else()
            # fall back to libstdc++
            target_compile_options(${target} PRIVATE -stdlib=libstdc++)
            target_link_options(${target} PRIVATE -stdlib=libstdc++)
        endif()
    endif()

    if(APPLE)
        # first check Apple since Apple is also a kind of Unix
        if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND add_apple_asan MATCHES true)
            if(USE_ASAN)
                target_compile_options(${target} PRIVATE "$<${debug_mode}:-fsanitize=address,undefined>")
                target_link_options(${target} PRIVATE "$<${debug_mode}:-fsanitize=address,undefined>")
            endif()
        endif()
    elseif(UNIX)
        if(USE_ASAN)
            # check leaks on Linux since macOS does not support the leaks sanitizer yet
            # leaks sanitizer is enabled by default on Linux, so we do not need to enable it explicitly
            target_compile_options(${target} PRIVATE "$<${debug_mode}:-fsanitize=address,undefined>")
            target_link_options(${target} PRIVATE "$<${debug_mode}:-fsanitize=address,undefined>")
        elseif(USE_MSAN)
            # use semi-colons instead of spaces to separate arguments
            # it is recommended to quote generator expressions in order to avoid unintentional splitting
            target_compile_options(${target} PRIVATE "$<${debug_mode}:-fsanitize=memory,undefined;-fsanitize-recover=memory,undefined;-fsanitize-memory-track-origins>")
            target_link_options(${target} PRIVATE "$<${debug_mode}:-fsanitize=memory,undefined;-fsanitize-recover=memory,undefined;-fsanitize-memory-track-origins;-Wl,-rpath,tools/llvm-project/build/lib>")
        endif()
    endif()
endfunction()
include(cmake/Options.cmake)
include(cmake/CompilerFlags.cmake)
include(cmake/CopyHelper.cmake)

###############################################################################

# external dependencies with FetchContent
include(FetchContent)
#
# set(FETCHCONTENT_QUIET OFF)
# set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
#
# NOTE: Also update env vars used for caching in
#         - .github/actions/configure-cmake/action.yml
#         - .github/workflows/cmake.yml
 FetchContent_Declare(
         googletest
         URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
 )
#
FetchContent_MakeAvailable(googletest)

###############################################################################

# external dependencies with find_package

# find_package(Threads REQUIRED)

###############################################################################

# NOTE: update executable name in .github/workflows/cmake.yml:25 when changing name here
# Support both main.cpp and main.cu
if(EXISTS "${CMAKE_SOURCE_DIR}/main.cu" AND CUDA_ENABLED)
    add_executable(${MAIN_EXECUTABLE_NAME} "${CMAKE_SOURCE_DIR}/main.cu" ${SOURCES})
    set_target_properties(${MAIN_EXECUTABLE_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        LINKER_LANGUAGE CUDA
    )
    # Link CUDA runtime
    target_link_libraries(${MAIN_EXECUTABLE_NAME} PRIVATE CUDA::cudart)
elseif(EXISTS "${CMAKE_SOURCE_DIR}/main.cpp")
    add_executable(${MAIN_EXECUTABLE_NAME} "${CMAKE_SOURCE_DIR}/main.cpp" ${SOURCES})
elseif(EXISTS "${CMAKE_SOURCE_DIR}/main.cu" AND NOT CUDA_ENABLED)
    message(FATAL_ERROR "This project requires CUDA but CUDA was not found. Please install CUDA toolkit.")
else()
    message(FATAL_ERROR "No main.cpp or main.cu found in project root.")
endif()

# Disable warnings-as-errors for CUDA builds to avoid NVCC host warnings.
if(CUDA_ENABLED AND PROJECT_WARNINGS_AS_ERRORS)
    message(STATUS "Warnings-as-errors disabled for CUDA build")
    set(PROJECT_WARNINGS_AS_ERRORS OFF)
endif()

# NOTE: Add all defined targets (e.g. executables, libraries, etc. )
# NOTE: RUN_SANITIZERS is optional, if it's not present it will default to true
set(RUN_SANITIZERS TRUE)
if(CUDA_ENABLED)
    set(RUN_SANITIZERS FALSE)
endif()
set_compiler_flags(RUN_SANITIZERS ${RUN_SANITIZERS} TARGET_NAMES ${MAIN_EXECUTABLE_NAME})
# set_compiler_flags(TARGET_NAMES ${MAIN_EXECUTABLE_NAME} ${FOO} ${BAR})
# where ${FOO} and ${BAR} represent additional executables or libraries
# you want to compile with the set compiler flags

###############################################################################

# target_include_directories(${MAIN_EXECUTABLE_NAME} SYSTEM PRIVATE ext/<SomeHppLib>/include)
# target_include_directories(${MAIN_EXECUTABLE_NAME} SYSTEM PRIVATE ${<SomeLib>_SOURCE_DIR}/include)
# target_link_directories(${MAIN_EXECUTABLE_NAME} PRIVATE ${<SomeLib>_BINARY_DIR}/lib)
# target_link_libraries(${MAIN_EXECUTABLE_NAME} <SomeLib>)

###############################################################################

# copy binaries to "bin" folder; these are uploaded as artifacts on each release
# DESTINATION_DIR is set as "bin" in cmake/Options.cmake:6
install(TARGETS ${MAIN_EXECUTABLE_NAME} DESTINATION ${DESTINATION_DIR})
if(APPLE)
    install(FILES launcher.command DESTINATION ${DESTINATION_DIR})
endif()

###############################################################################
#enable_testing()
#
#add_executable(
#        test_oop
#        ${HEADERS}
#        ${TESTS}
#        ${SOURCES}
#)
#
#target_link_libraries(
#        test_oop
#        GTest::gtest_main
#)
#
#include(GoogleTest)
#gtest_discover_tests(test_oop)
copy_files(FILES tastatura.txt COPY_TO_DESTINATION TARGET_NAME ${MAIN_EXECUTABLE_NAME})
# copy_files(FILES tastatura.txt config.json DIRECTORY images sounds COPY_TO_DESTINATION TARGET_NAME ${MAIN_EXECUTABLE_NAME})
# copy_files(DIRECTORY images sounds COPY_TO_DESTINATION TARGET_NAME ${MAIN_EXECUTABLE_NAME})
