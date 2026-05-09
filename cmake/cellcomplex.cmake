if (TARGET cellcomplex::cellcomplex)
    return()
endif()

include(CPM)
CPMAddPackage(
    NAME cellcomplex
    GITHUB_REPOSITORY duxingyi-charles/cell-complex
    GIT_TAG 3c37ff463468d434251d057c72d9ba2f78c4db93
)
