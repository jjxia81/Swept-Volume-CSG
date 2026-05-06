if (TARGET cellcomplex::cellcomplex)
    return()
endif()

include(CPM)
CPMAddPackage(
    NAME cellcomplex
    GITHUB_REPOSITORY duxingyi-charles/cell-complex
    GIT_TAG 512cafa46a6b1dce7e1e63d18dafe3bf3bd9a18c
)
