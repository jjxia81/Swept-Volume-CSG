if (TARGET cellcomplex::cellcomplex)
    return()
endif()

include(CPM)
CPMAddPackage(
    NAME cellcomplex
    GITHUB_REPOSITORY qnzhou/cell-complex
    GIT_TAG main
)
