if (TARGET cellcomplex::cellcomplex)
    return()
endif()

include(CPM)
CPMAddPackage(
    NAME cellcomplex
    GITHUB_REPOSITORY duxingyi-charles/cell-complex
    GIT_TAG 5841acb9ccf9922421529d620e98d7e6dafdb367
)
