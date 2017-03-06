include_directories (. ${PROJECT_SOURCE_DIR}/src/)
add_executable (learn2compose_snli
	snli.cc
	snli_corpus.cc
	snli_corpus.h
	snli_model.cc
	snli_model.h
	)

if (MSVC)
	target_link_libraries (learn2compose_snli dynet common ${Boost_LIBRARIES} ${LIBS})
else()
	target_link_libraries (learn2compose_snli dynet common ${Boost_LIBRARIES} ${LIBS} z)
endif()
