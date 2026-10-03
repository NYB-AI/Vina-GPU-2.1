void get_heavy_atom_movable_coords(output_type_cl* tmp, const __global m_ro_cl* restrict m, __global float* m_soa_buffer, int thread_id, int total_threads, __global ligand_atom_coords_cl* coords) {
	int counter = 0;
	for (int i = 0; i < m->m_num_movable_atoms; i++) {
		if (m->atoms[i].types[0] != EL_TYPE_H) {
			for (int j = 0; j < 3; j++)coords->coords[counter][j] = m_soa_buffer[SOA_IDX(M_COORDS_OFFSET, i, j, 3, thread_id, total_threads)];
			counter++;
		}
		else {
			//printf("\n kernel2: removed H atom coords in get_heavy_atom_movable_coords()!");
		}
	}
	//assign 0 for others
	for (int i = counter; i < MAX_NUM_OF_ATOMS; i++) {
		for (int j = 0; j < 3; j++)coords->coords[i][j] = 0;
	}
}


//Generate a random number according to step
float generate_n(__constant float* pi_map, const int step) {
	return fabs(pi_map[step]) / M_PI;
}

bool metropolis_accept(float old_f, float new_f, float temperature, float n) {
	if (new_f < old_f)return true;
	const float acceptance_probability = exp((old_f - new_f) / temperature);
	bool res = n < acceptance_probability;
	return n < acceptance_probability;
}

__kernel
void kernel2(
				const		__global		output_type_cl*			ric,
							__global const	m_ro_cl* restrict		mg,
							__constant		pre_cl*					pre,
							__constant		grids_cl*				grids,
							__constant		random_maps*			random_maps,
							__global		ligand_atom_coords_cl*	coords,
							__global		output_type_cl*			results,
				const		__global		mis_cl*					mis,
				const						int						torsion_size,
				const						int						search_depth,
				const						int						max_bfgs_steps,
				const 						int						rilc_bfgs_enable,
							__global		float*					m_soa_buffer
) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gs = get_global_size(0);
	int gl = get_global_linear_id();

	for (int gll = gl;
			 gll < mis->thread;
			 gll += mis->total_wi
		)
	{
		float best_e = INFINITY;
		output_type_cl best_out;


		output_type_cl tmp = ric[gll];

		change_cl g;

		output_type_cl candidate;

		for (int step = 0; step < search_depth; step++) {
			candidate = tmp;

			int map_index = (step + gll * search_depth) % MAX_NUM_OF_RANDOM_MAP;
			mutate_conf_cl(	map_index,
							&candidate,
							random_maps->int_map,
							random_maps->sphere_map,
							random_maps->pi_map,
							mg->ligand.begin,
							mg->ligand.end,
							mg->atoms,
							m_soa_buffer, gll, mis->thread,
							mis->epsilon_fl,
							mis->mutation_amplitude,
							torsion_size
			);
			if (rilc_bfgs_enable==1){
				rilc_bfgs(	&candidate,
						&g,
						mg, m_soa_buffer, gll, mis->thread,
						pre,
						grids,
						mis,
						torsion_size,
						max_bfgs_steps
				);
			}else{
				bfgs(	&candidate,
						&g,
						mg, m_soa_buffer, gll, mis->thread,
						pre,
						grids,
						mis,
						torsion_size,
						max_bfgs_steps
				);
			}
			
			float n = generate_n(random_maps->pi_map, map_index);

			if (step == 0 || metropolis_accept(tmp.e, candidate.e, 1.2, n)) {

				tmp = candidate;

				set(&tmp, mg, m_soa_buffer, gll, mis->thread,
					mis->epsilon_fl);

				if (tmp.e < best_e) {
					if (rilc_bfgs_enable==1){
						rilc_bfgs(	&tmp,
								&g,
								mg, m_soa_buffer, gll, mis->thread,
								pre,
								grids,
								mis,
								torsion_size,
								max_bfgs_steps
						);
					}else{
						bfgs(	&tmp,
								&g,
								mg, m_soa_buffer, gll, mis->thread,
								pre,
								grids,
								mis,
								torsion_size,
								max_bfgs_steps
						);
					}

					// set
					if (tmp.e < best_e) {
						set(&tmp, mg, m_soa_buffer, gll, mis->thread,
							mis->epsilon_fl);

						best_out = tmp;
						best_e = tmp.e;
					}

				}
				
				if (step == 0 && best_e == INFINITY) {
					best_out = tmp;
				}
			}
		}

		// write the best conformation back to CPU
		results[gll] = best_out;
		set(&best_out, mg, m_soa_buffer, gll, mis->thread, mis->epsilon_fl);
		get_heavy_atom_movable_coords(&best_out, mg, m_soa_buffer, gll, mis->thread, &coords[gll]);
	}
}