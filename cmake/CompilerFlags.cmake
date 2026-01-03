include(cmake/CustomStdlibAndSanitizers.cmake)

# target definitions

function(set_compiler_flags)
    set(multiValueArgs TARGET_NAMES)
    set(oneValueArgs RUN_SANITIZERS)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${oneValueArgs}" "${multiValueArgs}")

    if(NOT DEFINED ARG_RUN_SANITIZERS)
        set(ARG_RUN_SANITIZERS TRUE)
    endif()

    # iterate over all specified targets
    foreach (TARGET_NAME IN LISTS ARG_TARGET_NAMES)
        if(GITHUB_ACTIONS)
            message("NOTE: GITHUB_ACTIONS defined")
            target_compile_definitions(${TARGET_NAME} PRIVATE GITHUB_ACTIONS)
        endif()

        ###############################################################################

        if(PROJECT_WARNINGS_AS_ERRORS)
            set_property(TARGET ${TARGET_NAME} PROPERTY COMPILE_WARNING_AS_ERROR ON)
        endif()

        # custom compiler flags
        message("Compiler: ${CMAKE_CXX_COMPILER_ID} version ${CMAKE_CXX_COMPILER_VERSION}")
        if(MSVC)
            target_compile_options(${TARGET_NAME} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/W4>
                $<$<COMPILE_LANGUAGE:CXX>:/permissive->
                $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
                $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
                $<$<COMPILE_LANGUAGE:CXX>:/wd4996>
                $<$<COMPILE_LANGUAGE:CXX>:/external:anglebrackets>
                $<$<COMPILE_LANGUAGE:CXX>:/external:W0>
                $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
                $<$<COMPILE_LANGUAGE:CXX>:/MP>
            )
        else()
            target_compile_options(${TARGET_NAME} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wall>
                $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                $<$<COMPILE_LANGUAGE:CXX>:-pedantic>
            )
        endif()

        ###############################################################################

        # sanitizers
        if("${ARG_RUN_SANITIZERS}" STREQUAL "TRUE")
            set_custom_stdlib_and_sanitizers(${TARGET_NAME} true)
        endif ()
    endforeach ()
endfunction()
