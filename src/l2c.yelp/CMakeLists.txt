include_directories (. ${PROJECT_SOURCE_DIR}/src/)
add_executable (learn2compose_yelp
	yelp.cc
	yelp_corpus.cc
	yelp_corpus.h
	yelp_model.cc
	yelp_model.h
	)

if (MSVC)
	target_link_libraries (learn2compose_yelp dynet common ${Boost_LIBRARIES} ${LIBS})
else()
	target_link_libraries (learn2compose_yelp dynet common ${Boost_LIBRARIES} ${LIBS} z)
endif()
