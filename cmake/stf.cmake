# if (TARGET stf::stf)
#     return()
# endif()

# include(CPM)
# CPMAddPackage(
#     NAME stf
#     GITHUB_REPOSITORY adobe-research/space-time-functions
#     GIT_TAG b4c4a7e54400a2fa0389b2b738301e472c06282e
# )


if (TARGET stf::stf)
    return()
endif()

include(CPM)
CPMAddPackage(
    NAME stf
    GITHUB_REPOSITORY jjxia81/space-time-functions
    GIT_TAG main
    OPTIONS
        "STF_BUILD_TESTS OFF"
        "STF_YAML_PARSER ON"
        "STF_PYTHON_BINDING OFF"
    # GIT_TAG csg_dynamic
)
