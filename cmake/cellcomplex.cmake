if (TARGET cellcomplex::cellcomplex)
    return()
endif()

include(CPM)
CPMAddPackage(
    NAME cellcomplex
    GITHUB_REPOSITORY duxingyi-charles/cell-complex
    GIT_TAG 7073c540bc245c5537c71531fe4c792e5b9e3c26
)
