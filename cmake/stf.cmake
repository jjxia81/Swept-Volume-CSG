if (TARGET stf::stf)
    return()
endif()

include(CPM)
CPMAddPackage(
    NAME stf
    GITHUB_REPOSITORY adobe-research/space-time-functions
    GIT_TAG 2c641bd5a15db28cfead3eaa0f5f3421e6b3ea8b
)
