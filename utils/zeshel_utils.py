
MAX_ENT_LENGTH = 128
MAX_MENT_LENGTH = 128
MAX_PAIR_LENGTH = 256

N_ENTS_ZESHEL  = {
	"lego":10076,
	"star_trek":34430,
	"forgotten_realms":15603,
	"yugioh":10031,
	"american_football":31929,
	"fallout":16992,
	"pro_wrestling":10133,
	"military":104520,
	"doctor_who":40281,
	"final_fantasy":14044,
	"starwars":87056,
	"world_of_warcraft":27677,
	"coronation_street":17809,
	"muppets":21344,
	"ice_hockey":28684,
	"elder_scrolls":21712
}

N_MENTS_ZESHEL  = {
	"lego":1199,
	"star_trek":4227,
	"forgotten_realms":1200,
	"yugioh":3374,
	"american_football":3898,
	"fallout":3286,
	"pro_wrestling":1392,
	"military":13063,
	"doctor_who":8334,
	"final_fantasy":6041,
	"starwars":11824,
	"world_of_warcraft":1437,
	"coronation_street":1464,
	"muppets":2028,
	"ice_hockey":2233,
	"elder_scrolls":4275
}


def get_zeshel_world_info():
	train_worlds =  ["american_football", "doctor_who", "fallout", "final_fantasy", "military", "pro_wrestling",
					 "starwars", "world_of_warcraft"]
	test_worlds = ["forgotten_realms", "lego", "star_trek", "yugioh"]
	valid_worlds = ["coronation_street", "elder_scrolls", "ice_hockey", "muppets"]
	
	worlds = [("test",w) for w in test_worlds]
	worlds += [("train",w) for w in train_worlds]
	worlds += [("valid",w) for w in valid_worlds]
	
	return worlds


def get_dataset_info(data_dir, res_dir, worlds, n_ment=100):
	DATASETS = {
		world: {
			"ment_file": f"{data_dir}/processed/{world_type}_worlds/{world}_mentions.jsonl",
			"ent_file":f"{data_dir}/documents/{world}.json",
			"ent_tokens_file":f"{data_dir}/tokenized_entities/{world}_128_bert_base_uncased.npy",
		}
		for world_type, world in worlds
	}
	
	if res_dir is not None:
		# CrossEncoder score files for some domains/worlds
		n_ents = N_ENTS_ZESHEL
		domains = list(n_ents.keys())
		n_ments = N_MENTS_ZESHEL if n_ment is None else {domain:n_ment for domain in N_MENTS_ZESHEL}
		for domain in domains:
			filename_w_suff =  f"{res_dir}/{domain}/ment_to_ent_scores_n_m_{n_ments[domain]}_n_e_{n_ents[domain]}_all_layers_False.pkl"
			DATASETS[domain]["crossenc_ment_to_ent_scores"] =  filename_w_suff
			DATASETS[domain]["crossenc_ment_and_ent_embeds"] =  f"{res_dir}/{domain}/ment_and_ent_embeds_n_m_{n_ments[domain]}_n_e_{n_ents[domain]}_all_layers_False.pkl"

		
	return DATASETS
