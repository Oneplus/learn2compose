include_directories ()
add_library (common
	corpus.h
	corpus.cc
	alphabet.cc
	alphabet.h
	logging.cc
	logging.h
	layer.cc
	layer.h
	system.cc
	system.h
	reinforce.cc
	reinforce.h
	trainer_utils.cc
	trainer_utils.h
	)

add_executable (learn2compose_snli
	snli.cc
	snli_corpus.cc
	snli_corpus.h
	snli_model.cc
	snli_model.h
	)

add_executable (learn2compose_sst
	sst.cc
	sst_corpus.cc
	sst_corpus.h
	sst_model.cc
	sst_model.h
	)
	
add_executable (learn2compose_yelp
	yelp.cc
	yelp_corpus.cc
	yelp_corpus.h
	yelp_model.cc
	yelp_model.h
	)

if (MSVC)
	target_link_libraries (learn2compose_snli dynet common ${Boost_LIBRARIES} ${LIBS})
	target_link_libraries (learn2compose_sst dynet common ${Boost_LIBRARIES} ${LIBS})
	target_link_libraries (learn2compose_yelp dynet common ${Boost_LIBRARIES} ${LIBS})
else()
	target_link_libraries (learn2compose_snli dynet common ${Boost_LIBRARIES} ${LIBS} z)
	target_link_libraries (learn2compose_sst dynet common ${Boost_LIBRARIES} ${LIBS} z)
	target_link_libraries (learn2compose_yelp dynet common ${Boost_LIBRARIES} ${LIBS} z)
endif()
