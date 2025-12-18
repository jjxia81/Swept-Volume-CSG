if (TARGET mtetcol::mtetcol)
    return()
endif()

include(CPM)
CPMAddPackage(
  NAME mtetcol
  GITHUB_REPOSITORY adobe-research/mtetcol
  GIT_TAG c0e7253dd59f5f20b37f0bff541b86d226b83ac5
)

