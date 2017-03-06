include_directories (. ${PROJECT_SOURCE_DIR}/src/)
add_executable (learn2compose_sst
	sst.cc
	sst_corpus.cc
	sst_corpus.h
	sst_model.cc
	sst_model.h
	)

if (MSVC)
	target_link_libraries (learn2compose_sst dynet common ${Boost_LIBRARIES} ${LIBS})
else()
	target_link_libraries (learn2compose_sst dynet common ${Boost_LIBRARIES} ${LIBS} z)
endif()
