#include "cache.h"
#include "wrapcl.h"
#include "monte_carlo.h"
#include "coords.h"
#include "mutate.h"
#include "quasi_newton.h"
#include "parallel_mc.h"
#include "szv_grid.h"
#include <thread>
#include <boost/progress.hpp>

#include "commonMacros.h"
#include "wrapcl.h"
#include "random.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <stdio.h>
#include <cstdlib>
#include <unistd.h>
#include <algorithm>

namespace {
// Every replica starts from the same state, so each SoA row (one scalar field
// across all `thread` replicas) holds a single constant. Fill it contiguously
// rather than scattering one float per replica at stride `thread`.
inline void soa_fill(float* soa, int offset, int i, int j, int dim_j, int thread, float v) {
	float* row = soa + (size_t)(offset + i * dim_j + j) * (size_t)thread;
	std::fill(row, row + thread, v);
}

inline std::string vina_gpu_log_path() {
	const char* env = std::getenv("VINA_GPU_LOG");
	if (env) return std::string(env);
	return std::string("gpu_runtime.log.") + std::to_string(getpid());
}
}

using namespace std;
volatile enum { FINISH, DOCKING, ABORT} status;
void print_process() {
	int count = 0;
	printf("\n");
	do
	{
#ifdef WIN32
		Sleep(100);
#else
		sleep(1);
#endif
		printf("\rPerform docking|");
		for (int i = 0; i < count; i++)printf(" ");
		printf("=======");
		for (int i = 0; i < 30 - count; i++)printf(" ");
		printf("|"); fflush(stdout);

		count++;
		count %= 30;
	} while (status == DOCKING);
	if (status == FINISH) {
		printf("\rPerform docking|");
		for (int i = 0; i < 16; i++)printf("=");
		printf("done");
		for (int i = 0; i < 17; i++)printf("=");
		printf("|\n"); fflush(stdout);
	}
	else if (status == ABORT) {
		printf("\rPerform docking|");
		for (int i = 0; i < 16; i++)printf("=");
		printf("error");
		for (int i = 0; i < 16; i++)printf("=");
		printf("|\n"); fflush(stdout);
	}
}

std::vector<output_type> cl_to_vina(output_type_cl result_ptr[], 
									ligand_atom_coords_cl result_coords_ptr[],
									int thread, int lig_torsion_size) 
{
	std::vector<output_type> results_vina;
	int num_atoms;
	for (int i = 0; i < thread; i++) {
		output_type_cl tmp = result_ptr[i];
		ligand_atom_coords_cl tmp_coords = result_coords_ptr[i];
		conf tmp_c;
		tmp_c.ligands.resize(1);
		// Position
		for (int j = 0; j < 3; j++)tmp_c.ligands[0].rigid.position[j] = tmp.position[j];
		// Orientation
		qt q(tmp.orientation[0], tmp.orientation[1], tmp.orientation[2], tmp.orientation[3]);
		tmp_c.ligands[0].rigid.orientation = q;
		output_type tmp_vina(tmp_c, tmp.e);
		// torsion
		for (int j = 0; j < lig_torsion_size; j++)tmp_vina.c.ligands[0].torsions.push_back(tmp.lig_torsion[j]);
		// coords
		for (int j = 0; j < MAX_NUM_OF_ATOMS; j++) {
			vec v_tmp(tmp_coords.coords[j][0], tmp_coords.coords[j][1], tmp_coords.coords[j][2]);
			if ((v_tmp[0] != 0 || v_tmp[1] != 0) || (v_tmp[2] != 0)) tmp_vina.coords.push_back(v_tmp);
		}
		results_vina.push_back(tmp_vina);
		if (i == 0)num_atoms = tmp_vina.coords.size();
		if(num_atoms != tmp_vina.coords.size()){throw std::runtime_error("atom coords not match!");}
	}
	return results_vina;
}

void main_procedure_cl(cache& c, const std::vector<model>& ms,  const precalculate& p, const parallel_mc par,
	const vec& corner1, const vec& corner2, const int seed, std::vector<output_container>& outs, std::string opencl_binary_path,
	const std::vector<std::vector<std::string>> ligand_names, const int rilc_bfgs, const int gpu_device_id) {

/**************************************************************************/
/***************************    OpenCL Init    ****************************/
/**************************************************************************/

	cl_int err;
	cl_platform_id* platforms;
	cl_device_id* devices;
	cl_context context;
	cl_command_queue queue;
	cl_int gpu_platform_id = 1;
	
	printf("\nInitializing OpenCL with GPU device ID: %d", gpu_device_id);
	printf("\nIf you have multiple GPUs, use --gpu_device <id> parameter to select a specific GPU");
	
	SetupPlatform(&platforms, &gpu_platform_id);
	
	// Get the number of devices on the selected platform
	cl_uint num_devices = 0;
	err = clGetDeviceIDs(platforms[gpu_platform_id], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices); checkErr(err);
	
	// Set up device list
	SetupDevice(platforms, &devices, gpu_platform_id);
	
	// Create context with all available devices
	SetupContext(platforms, devices[gpu_device_id], &context, gpu_platform_id);
	
	// Try to set up the queue with the requested device ID
	try {
		SetupQueue(&queue, context, devices, gpu_device_id);
	} catch (const std::exception& e) {
		printf("\nError setting up GPU device %d: %s", gpu_device_id, e.what());
		printf("\nFalling back to device 0");
		SetupQueue(&queue, context, devices, 0);
	}
	
	cl_program programs[2];

	//printf("\nSearch depth is set to %d", par.mc.search_depth);
	printf("\nUsing random seed: %d", seed);

#ifdef BUILD_KERNEL_FROM_SOURCE
	const std::string default_work_path = ".";
	const std::string include_path = default_work_path + "/OpenCL/inc"; //FIX it
	const std::string addtion = "";

	printf("\n\nBuild kernel 1 from source"); fflush(stdout);
	char* program1_file_n[NUM_OF_FILES_KERNEL_1];
	size_t program1_size_n[NUM_OF_FILES_KERNEL_1];
	std::string file1_paths[NUM_OF_FILES_KERNEL_1] = {	default_work_path + "/OpenCL/src/kernels/code_head.cl",
														//default_work_path + "/OpenCL/src/kernels/mutate_conf.cl",
														//default_work_path + "/OpenCL/src/kernels/matrix.cl",
														//default_work_path + "/OpenCL/src/kernels/quasi_newton.cl",
														default_work_path + "/OpenCL/src/kernels/kernel1.cl" }; // The order of files is important!

	read_n_file(program1_file_n, program1_size_n, file1_paths, NUM_OF_FILES_KERNEL_1);
	std::string final_file;
	size_t final_size = NUM_OF_FILES_KERNEL_1 - 1; // count '\n'
	for (int i = 0; i < NUM_OF_FILES_KERNEL_1; i++) {
		if (i == 0) final_file = program1_file_n[0];
		else final_file = final_file + '\n' + (std::string)program1_file_n[i];
		final_size += program1_size_n[i];
	}
	const char* final_files1_char = final_file.data();

	programs[0] = clCreateProgramWithSource(context, 1, (const char**)&final_files1_char, &final_size, &err); checkErr(err);
	SetupBuildProgramWithSource(programs[0], NULL, devices, include_path, addtion);
	SaveProgramToBinary(programs[0], "Kernel1_Opt.bin");


	printf("\nBuild kernel 2 from source"); fflush(stdout);
	char* program2_file_n[NUM_OF_FILES_KERNEL_2];
	size_t program2_size_n[NUM_OF_FILES_KERNEL_2];
	std::string file2_paths[NUM_OF_FILES_KERNEL_2] = { default_work_path + "/OpenCL/src/kernels/code_head.cl",
													   default_work_path + "/OpenCL/src/kernels/mutate_conf.cl",
													   default_work_path + "/OpenCL/src/kernels/matrix.cl",
													   default_work_path + "/OpenCL/src/kernels/quasi_newton.cl",
													   default_work_path + "/OpenCL/src/kernels/kernel2.cl" }; // The order of files is important!

	read_n_file(program2_file_n, program2_size_n, file2_paths, NUM_OF_FILES_KERNEL_2);
	final_size = NUM_OF_FILES_KERNEL_2 - 1; // count '\n'
	for (int i = 0; i < NUM_OF_FILES_KERNEL_2; i++) {
		if (i == 0) final_file = program2_file_n[0];
		else final_file = final_file + '\n' + (std::string)program2_file_n[i];
		final_size += program2_size_n[i];
	}
	const char* final_files2_char = final_file.data();

	programs[1] = clCreateProgramWithSource(context, 1, (const char**)&final_files2_char, &final_size, &err); checkErr(err);
	SetupBuildProgramWithSource(programs[1], NULL, devices, include_path, addtion);
	SaveProgramToBinary(programs[1], "Kernel2_Opt.bin");
#endif

	// Try to load pre-compiled binaries first
	try {
		printf("\nAttempting to load pre-compiled OpenCL binaries for device %d...", gpu_device_id);
		programs[0] = SetupBuildProgramWithBinary(context, devices[gpu_device_id], (opencl_binary_path + std::string("/Kernel1_Opt.bin")).c_str());
		programs[1] = SetupBuildProgramWithBinary(context, devices[gpu_device_id], (opencl_binary_path + std::string("/Kernel2_Opt.bin")).c_str());
	} 
	catch (const std::exception& e) {
		printf("\nError loading pre-compiled binaries: %s", e.what());
		printf("\nThis may happen when using a different GPU than the one for which the binaries were compiled.");
		printf("\nRebuilding kernels from source for the current device...");

		// Fall back to building from source
		const std::string default_work_path = ".";
		const std::string include_path = default_work_path + "/OpenCL/inc";
		const std::string addtion = "";

		// Build kernel 1
		printf("\nBuilding kernel 1 from source..."); fflush(stdout);
		char* program1_file_n[NUM_OF_FILES_KERNEL_1];
		size_t program1_size_n[NUM_OF_FILES_KERNEL_1];
		std::string file1_paths[NUM_OF_FILES_KERNEL_1] = {
			default_work_path + "/OpenCL/src/kernels/code_head.cl",
			default_work_path + "/OpenCL/src/kernels/kernel1.cl"
		};

		read_n_file(program1_file_n, program1_size_n, file1_paths, NUM_OF_FILES_KERNEL_1);
		std::string final_file;
		size_t final_size = NUM_OF_FILES_KERNEL_1 - 1;
		for (int i = 0; i < NUM_OF_FILES_KERNEL_1; i++) {
			if (i == 0) final_file = program1_file_n[0];
			else final_file = final_file + '\n' + (std::string)program1_file_n[i];
			final_size += program1_size_n[i];
		}
		const char* final_files1_char = final_file.data();

		programs[0] = clCreateProgramWithSource(context, 1, (const char**)&final_files1_char, &final_size, &err); checkErr(err);
		SetupBuildProgramWithSource(programs[0], NULL, devices, include_path, addtion);
		SaveProgramToBinary(programs[0], "Kernel1_Opt.bin");

		// Build kernel 2
		printf("\nBuilding kernel 2 from source..."); fflush(stdout);
		char* program2_file_n[NUM_OF_FILES_KERNEL_2];
		size_t program2_size_n[NUM_OF_FILES_KERNEL_2];
		std::string file2_paths[NUM_OF_FILES_KERNEL_2] = {
			default_work_path + "/OpenCL/src/kernels/code_head.cl",
			default_work_path + "/OpenCL/src/kernels/mutate_conf.cl",
			default_work_path + "/OpenCL/src/kernels/matrix.cl",
			default_work_path + "/OpenCL/src/kernels/quasi_newton.cl",
			default_work_path + "/OpenCL/src/kernels/kernel2.cl"
		};

		read_n_file(program2_file_n, program2_size_n, file2_paths, NUM_OF_FILES_KERNEL_2);
		final_size = NUM_OF_FILES_KERNEL_2 - 1;
		for (int i = 0; i < NUM_OF_FILES_KERNEL_2; i++) {
			if (i == 0) final_file = program2_file_n[0];
			else final_file = final_file + '\n' + (std::string)program2_file_n[i];
			final_size += program2_size_n[i];
		}
		const char* final_files2_char = final_file.data();

		programs[1] = clCreateProgramWithSource(context, 1, (const char**)&final_files2_char, &final_size, &err); checkErr(err);
		SetupBuildProgramWithSource(programs[1], NULL, devices, include_path, addtion);
		SaveProgramToBinary(programs[1], "Kernel2_Opt.bin");
	}

	err = clUnloadPlatformCompiler(platforms[gpu_platform_id]); checkErr(err);
	//Set kernel arguments
	cl_kernel kernels[2];
	char kernel_name[][50] = { "kernel1","kernel2"};
	SetupKernel(kernels, programs, 2, kernel_name);

	size_t max_wg_size; // max work item within one work group
	size_t max_wi_size[3]; // max work item within each dimension(global)
	err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL); checkErr(err);
	err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * sizeof(size_t), &max_wi_size, NULL); checkErr(err);

	/**************************************************************************/
	/************************    Original Vina code    ************************/
	/**************************************************************************/
	sz nat = num_atom_types(c.atu);

	szv needed;
	for (int i = 0; i < nat; i++) {
		if (!c.grids[i].initialized()) {
			needed.push_back(i);
			c.grids[i].init(c.gd);
		}
	}

	flv affinities(needed.size());                

	grid& g = c.grids[needed.front()];              

	const fl cutoff_sqr = p.cutoff_sqr();         

	grid_dims gd_reduced = szv_grid_dims(c.gd);     
	szv_grid ig(ms[0], gd_reduced, cutoff_sqr); // use ms[0]           
	//szv_grid ig2(ms[1], gd_reduced, cutoff_sqr); // use ms[0]  

	for (int i = 0; i < 3; i++) {
		if (ig.m_init[i] != g.m_init[i]) {
			printf("m_init not equal!");
			exit(-1);
		}
		if (ig.m_range[i] != g.m_range[i]) {
			printf("m_range not equal!");
			exit(-1);
		}
	}

	vec authentic_v(1000, 1000, 1000); // FIXME? this is here to avoid max_fl/max_fl

	std::vector<conf_size> s;
	std::vector<output_type> tmps;
	for (int i = 0; i < ms.size(); i++) {
		s.push_back(ms[i].get_size());
		output_type tmp(s[i], 0);
		tmps.push_back(tmp);
	}
	//change g(s);
	
	//quasi_newton quasi_newton_par; const int quasi_newton_par_max_steps = par.mc.ssd_par.evals;
	/**************************************************************************/
	/************************    Kernel1    ***********************/
	/**************************************************************************/
	rng generator(static_cast<rng::result_type>(seed));
	model m = ms[0];

	// Preparing protein atoms related data
	pa_cl* pa_ptr = (pa_cl*)malloc(sizeof(pa_cl));
	if(MAX_NUM_OF_PROTEIN_ATOMS <= m.grid_atoms.size()){throw std::runtime_error("pocket too large!");}
	for (int i = 0; i < m.grid_atoms.size(); i++) {
		pa_ptr->atoms[i].types[0] = m.grid_atoms[i].el;
		pa_ptr->atoms[i].types[1] = m.grid_atoms[i].ad;
		pa_ptr->atoms[i].types[2] = m.grid_atoms[i].xs;
		pa_ptr->atoms[i].types[3] = m.grid_atoms[i].sy;
		for (int j = 0; j < 3; j++)pa_ptr->atoms[i].coords[j] = m.grid_atoms[i].coords.data[j];
	}
	size_t pa_size = sizeof(pa_cl);

	// Preaparing precalculated look up table related data
	pre_cl* pre_ptr = (pre_cl*)malloc(sizeof(pre_cl));
	pre_ptr->m_cutoff_sqr = p.cutoff_sqr();
	pre_ptr->factor = p.factor;
	pre_ptr->n = p.n;
	if(MAX_P_DATA_M_DATA_SIZE <= p.data.m_data.size()){throw std::runtime_error("LUT too large!");}
	for (int i = 0; i < p.data.m_data.size(); i++) {
		pre_ptr->m_data[i].factor = p.data.m_data[i].factor;
		if(FAST_SIZE != p.data.m_data[i].fast.size()){throw std::runtime_error("fast too large!");}
		if(SMOOTH_SIZE != p.data.m_data[i].smooth.size()){throw std::runtime_error("smooth too large!");}

		for (int j = 0; j < FAST_SIZE; j++) {
			pre_ptr->m_data[i].fast[j] = p.data.m_data[i].fast[j];
		}
		for (int j = 0; j < SMOOTH_SIZE; j++) {
			pre_ptr->m_data[i].smooth[j][0] = p.data.m_data[i].smooth[j].first;
			pre_ptr->m_data[i].smooth[j][1] = p.data.m_data[i].smooth[j].second;
		}
	}
	size_t pre_size = sizeof(pre_cl);

	// Preparing grid boundries
	gb_cl* gb_ptr = (gb_cl*)malloc(sizeof(gb_cl));
	gb_ptr->dims[0] = ig.m_data.dim0(); gb_ptr->dims[1] = ig.m_data.dim1(); gb_ptr->dims[2] = ig.m_data.dim2();
	for (int i = 0; i < 3; i++)gb_ptr->init[i] = ig.m_init.data[i];
	for (int i = 0; i < 3; i++)gb_ptr->range[i] = ig.m_range.data[i];
	size_t gb_size = sizeof(gb_cl);

	// Preparing atom relationship
	if(MAX_NUM_OF_ATOM_RELATION_COUNT < ig.m_data.m_data.size()){throw std::runtime_error("Relation too large! Define a large box (see readme) would help");}
	// assert(ig.m_data.m_i <= 10); assert(ig.m_data.m_j <= 10); assert(ig.m_data.m_k <= 10);
	ar_cl* ar_ptr = (ar_cl*)malloc(sizeof(ar_cl));
	for (int i = 0; i < ig.m_data.m_data.size(); i++) {
		ar_ptr->relation_size[i] = ig.m_data.m_data[i].size();
		if(MAX_NUM_OF_ATOM_RELATION_COUNT < ar_ptr->relation_size[i]){throw std::runtime_error("Relation too large! Define a large box (see readme) would help");}
		for (int j = 0; j < ar_ptr->relation_size[i]; j++) {
			ar_ptr->relation[i][j] = ig.m_data.m_data[i][j];
		}
	}
	size_t ar_size = sizeof(ar_cl);

	// Preparing grid related data
	if(GRIDS_SIZE != c.grids.size()){throw std::runtime_error("grid_size has to be 17!");} // grid_size has to be 17
	grids_cl* grids_ptr = (grids_cl*)malloc(sizeof(grids_cl)); if (grids_ptr == nullptr) {throw std::runtime_error("Grid too large! Define a large box (see readme) would help");}
	grid* tmp_grid_ptr = &c.grids[0];

	grids_ptr->atu = c.atu; // atu
	grids_ptr->slope = c.slope; // slope
	int grids_front;
	for (int i = GRIDS_SIZE - 1; i >= 0; i--) {
		for (int j = 0; j < 3; j++) {
			grids_ptr->grids[i].m_init[j] = tmp_grid_ptr[i].m_init[j];
			grids_ptr->grids[i].m_factor[j] = tmp_grid_ptr[i].m_factor[j];
			grids_ptr->grids[i].m_dim_fl_minus_1[j] = tmp_grid_ptr[i].m_dim_fl_minus_1[j];
			grids_ptr->grids[i].m_factor_inv[j] = tmp_grid_ptr[i].m_factor_inv[j];
		}
		if (tmp_grid_ptr[i].m_data.dim0() != 0) {
			grids_ptr->grids[i].m_i = tmp_grid_ptr[i].m_data.dim0(); if(MAX_NUM_OF_GRID_MI < grids_ptr->grids[i].m_i){throw std::runtime_error("MAX_NUM_OF_GRID_MI too small!  Define a large box (see readme) would help");}
			grids_ptr->grids[i].m_j = tmp_grid_ptr[i].m_data.dim1(); if(MAX_NUM_OF_GRID_MJ < grids_ptr->grids[i].m_j){throw std::runtime_error("MAX_NUM_OF_GRID_MJ too small!  Define a large box (see readme) would help");}
			grids_ptr->grids[i].m_k = tmp_grid_ptr[i].m_data.dim2(); if(MAX_NUM_OF_GRID_MK < grids_ptr->grids[i].m_k){throw std::runtime_error("MAX_NUM_OF_GRID_MK too small!  Define a large box (see readme) would help");}
			grids_front = i;
		}
		else {
			grids_ptr->grids[i].m_i = 0;
			grids_ptr->grids[i].m_j = 0;
			grids_ptr->grids[i].m_k = 0;
		}
	}
	size_t grids_size = sizeof(grids_cl);

	// miscellaneous
	mis_cl* mis_ptr = (mis_cl*)malloc(sizeof(mis_cl));
	mis_ptr->needed_size = needed.size();
	mis_ptr->epsilon_fl = epsilon_fl;
	mis_ptr->cutoff_sqr = cutoff_sqr;
	mis_ptr->max_fl = max_fl;
	//mis_ptr->torsion_size = tmp.c.ligands[0].torsions.size();
	//mis_ptr->max_bfgs_steps = quasi_newton_par_max_steps;
	//mis_ptr->search_depth = par.mc.search_depth;
	mis_ptr->mutation_amplitude = par.mc.mutation_amplitude;
	// Right-size kernel2 launch to actual replica count, rounded up to a
	// warp-multiple (32). Old code launched a fixed grid sized to device max
	// regardless of par.mc.thread, which wasted SM occupancy on small jobs and
	// made multi-process co-execution unnecessarily expensive. The kernel's
	// gll stride loop already handles thread > total_wi correctly, so any
	// total_wi >= 32 is valid.
	{
		const int wg = 32;
		int wi = ((par.mc.thread + wg - 1) / wg) * wg;
		if (wi < wg) wi = wg;
		mis_ptr->total_wi = wi;
	}
	mis_ptr->thread = par.mc.thread;
	mis_ptr->ar_mi = ig.m_data.m_i;
	mis_ptr->ar_mj = ig.m_data.m_j;
	mis_ptr->ar_mk = ig.m_data.m_k;
	mis_ptr->grids_front = grids_front;
	for (int i = 0; i < 3; i++) mis_ptr->authentic_v[i] = authentic_v[i];
	for (int i = 0; i < 3; i++) mis_ptr->hunt_cap[i] = par.mc.hunt_cap[i];
	size_t mis_size = sizeof(mis_cl);

	float* needed_ptr = (float*)malloc(mis_ptr->needed_size * sizeof(float));
	for (int i = 0; i < mis_ptr->needed_size; i++)needed_ptr[i] = needed[i];

	cl_mem pre_gpu;
	CreateDeviceBuffer(&pre_gpu, CL_MEM_READ_ONLY, pre_size, context);
	err = clEnqueueWriteBuffer(queue, pre_gpu, false, 0, pre_size, pre_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem pa_gpu;
	CreateDeviceBuffer(&pa_gpu, CL_MEM_READ_ONLY, pa_size, context);
	err = clEnqueueWriteBuffer(queue, pa_gpu, false, 0, pa_size, pa_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem gb_gpu;
	CreateDeviceBuffer(&gb_gpu, CL_MEM_READ_ONLY, gb_size, context);
	err = clEnqueueWriteBuffer(queue, gb_gpu, false, 0, gb_size, gb_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem ar_gpu;
	CreateDeviceBuffer(&ar_gpu, CL_MEM_READ_ONLY, ar_size, context);
	err = clEnqueueWriteBuffer(queue, ar_gpu, false, 0, ar_size, ar_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem grids_gpu;
	CreateDeviceBuffer(&grids_gpu, CL_MEM_READ_WRITE, grids_size, context);
	err = clEnqueueWriteBuffer(queue, grids_gpu, false, 0, grids_size, grids_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem needed_gpu;
	CreateDeviceBuffer(&needed_gpu, CL_MEM_READ_WRITE, mis_ptr->needed_size * sizeof(float), context);
	err = clEnqueueWriteBuffer(queue, needed_gpu, false, 0, mis_ptr->needed_size * sizeof(float), needed_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem mis_gpu;
	CreateDeviceBuffer(&mis_gpu, CL_MEM_READ_ONLY, mis_size, context);
	err = clEnqueueWriteBuffer(queue, mis_gpu, false, 0, mis_size, mis_ptr, 0, NULL, NULL); checkErr(err);
	
	clFinish(queue);

	SetKernelArg(kernels[0], 0, sizeof(cl_mem), &pre_gpu);
	SetKernelArg(kernels[0], 1, sizeof(cl_mem), &pa_gpu);
	SetKernelArg(kernels[0], 2, sizeof(cl_mem), &gb_gpu);
	SetKernelArg(kernels[0], 3, sizeof(cl_mem), &ar_gpu);
	SetKernelArg(kernels[0], 4, sizeof(cl_mem), &grids_gpu);
	SetKernelArg(kernels[0], 5, sizeof(cl_mem), &mis_gpu);
	SetKernelArg(kernels[0], 6, sizeof(cl_mem), &needed_gpu);
	SetKernelArg(kernels[0], 7, sizeof(int), &c.atu);
	SetKernelArg(kernels[0], 8, sizeof(int), &nat);

	status = DOCKING;
# ifdef NDEBUG
	std::thread console_thread(print_process);
# endif

	cl_event kernel1;
	// Cover every grid point. A fixed 128^3 launch silently left the grid
	// layers beyond 128 points (boxes wider than ~47.6 A at 0.375 A spacing)
	// uncomputed, i.e. zero energy. kernel1 returns early for x/y/z >= m_i/m_j/m_k.
	size_t kernel1_local_size[3] = { 4,4,4 };
	size_t kernel1_global_size[3] = {
		((size_t)grids_ptr->grids[grids_front].m_i + kernel1_local_size[0] - 1) / kernel1_local_size[0] * kernel1_local_size[0],
		((size_t)grids_ptr->grids[grids_front].m_j + kernel1_local_size[1] - 1) / kernel1_local_size[1] * kernel1_local_size[1],
		((size_t)grids_ptr->grids[grids_front].m_k + kernel1_local_size[2] - 1) / kernel1_local_size[2] * kernel1_local_size[2] };

	err = clEnqueueNDRangeKernel(queue, kernels[0], 3, 0, kernel1_global_size, kernel1_local_size, 0, NULL, &kernel1); checkErr(err);
	clWaitForEvents(1, &kernel1);

	free(pa_ptr);
	free(gb_ptr);
	free(ar_ptr);
	free(needed_ptr);
	free(pre_ptr);
	free(grids_ptr);

	err = clReleaseMemObject(pa_gpu);		checkErr(err);
	err = clReleaseMemObject(gb_gpu);		checkErr(err);
	err = clReleaseMemObject(ar_gpu);		checkErr(err);
	err = clReleaseMemObject(needed_gpu);	checkErr(err);

	err = clReleaseEvent(kernel1);			checkErr(err);
/**************************************************************************/
/************************    Kernel2    ***********************/
/**************************************************************************/
	// --- Pinned host buffers for kernel2 outputs (Tier 1d).
	// Allocate once outside the per-ligand loop since par.mc.thread is fixed.
	// Using CL_MEM_ALLOC_HOST_PTR + clEnqueueMapBuffer gives DMA-capable host
	// memory so clEnqueueReadBuffer below avoids an extra staging-buffer copy.
	const size_t pinned_results_bytes = par.mc.thread * sizeof(output_type_cl);
	const size_t pinned_coords_bytes  = par.mc.thread * sizeof(ligand_atom_coords_cl);
	cl_mem pinned_results_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
		pinned_results_bytes, NULL, &err); checkErr(err);
	cl_mem pinned_coords_buf  = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
		pinned_coords_bytes, NULL, &err); checkErr(err);
	output_type_cl* pinned_results_host = (output_type_cl*)clEnqueueMapBuffer(queue, pinned_results_buf,
		CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, pinned_results_bytes, 0, NULL, NULL, &err); checkErr(err);
	ligand_atom_coords_cl* pinned_coords_host = (ligand_atom_coords_cl*)clEnqueueMapBuffer(queue, pinned_coords_buf,
		CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, pinned_coords_bytes, 0, NULL, NULL, &err); checkErr(err);

	int num_ligands = ms.size();
	std::vector<random_maps*>			rand_maps_ptrs		(num_ligands);
	std::vector<output_type_cl*>		ric_ptrs			(num_ligands);
	std::vector<m_ro_cl*>					m_ptrs				(num_ligands);
	std::vector<ligand_atom_coords_cl*> result_coords_ptrs	(num_ligands);
	std::vector<output_type_cl*>		result_ptrs			(num_ligands);
	std::vector<int>					torsion_sizes		(num_ligands);

	std::vector<cl_mem> ric_gpus			(num_ligands);
	std::vector<cl_mem> m_gpus				(num_ligands);
	std::vector<cl_mem> m_soa_gpus			(num_ligands);
	std::vector<cl_mem> random_maps_gpus	(num_ligands);
	std::vector<cl_mem> result_coords_gpus	(num_ligands);
	std::vector<cl_mem> result_gpus			(num_ligands);

	std::vector<cl_event> ligands_events	(num_ligands);


	for (int ligand_count = 0; ligand_count < num_ligands; ligand_count++) {
		generator.seed(static_cast<rng::result_type>(seed));
# ifndef NDEBUG
		printf("\nDocking ligand: %s\n", ligand_names[ligand_count][0].c_str());
# endif
	try {
		model m = ms[ligand_count];
		if(m.atoms.size() >= MAX_NUM_OF_ATOMS){throw std::runtime_error("Ligand too large! The maximum number of atoms is " +
		 std::to_string(MAX_NUM_OF_ATOMS) + " and the ligand has " + std::to_string(m.atoms.size()) + " atoms.");}

		output_type tmp = tmps[ligand_count];
		torsion_sizes[ligand_count] = tmp.c.ligands[0].torsions.size();
		if (tmp.c.ligands[0].torsions.size() >= MAX_NUM_OF_LIG_TORSION) { throw std::runtime_error("Ligand too large! The maximum number of torsions is " +
		 std::to_string(MAX_NUM_OF_LIG_TORSION) + " and the ligand has " + std::to_string(tmp.c.ligands[0].torsions.size()) + " torsions."); }

		// Generate random maps
		rand_maps_ptrs[ligand_count] = (random_maps*)malloc(sizeof(random_maps));
		random_maps* rand_maps_ptr = rand_maps_ptrs[ligand_count];
		for (int i = 0; i < MAX_NUM_OF_RANDOM_MAP; i++) {
			rand_maps_ptr->int_map[i] = random_int(0, int(tmp.c.ligands[0].torsions.size()), generator);
			rand_maps_ptr->pi_map[i] = random_fl(-pi, pi, generator);
		}
		for (int i = 0; i < MAX_NUM_OF_RANDOM_MAP; i++) {
			vec rand_coords = random_inside_sphere(generator);
			for (int j = 0; j < 3; j++) {
				rand_maps_ptr->sphere_map[i][j] = rand_coords[j];
			}
		}
		size_t rand_maps_size = sizeof(*rand_maps_ptr);

		// Preparing random initial conformations
		ric_ptrs[ligand_count] = (output_type_cl*)malloc(par.mc.thread * sizeof(output_type_cl));
		output_type_cl* ric_ptr = ric_ptrs[ligand_count];
		for (int i = 0; i < par.mc.thread; i++) {
			tmp.c.randomize(corner1, corner2, generator);
			for (int j = 0; j < 3; j++)ric_ptr[i].position[j] = tmp.c.ligands[0].rigid.position[j];
			ric_ptr[i].orientation[0] = tmp.c.ligands[0].rigid.orientation.R_component_1();
			ric_ptr[i].orientation[1] = tmp.c.ligands[0].rigid.orientation.R_component_2();
			ric_ptr[i].orientation[2] = tmp.c.ligands[0].rigid.orientation.R_component_3();
			ric_ptr[i].orientation[3] = tmp.c.ligands[0].rigid.orientation.R_component_4();
			for (int j = 0; j < tmp.c.ligands[0].torsions.size(); j++)ric_ptr[i].lig_torsion[j] = tmp.c.ligands[0].torsions[j];
		}
		size_t ric_size = par.mc.thread * sizeof(output_type_cl);

		// Preparing m related data
		m_ptrs[ligand_count] = (m_ro_cl*)malloc(sizeof(m_ro_cl));
		m_ro_cl* m_ptr = m_ptrs[ligand_count];

		size_t soa_size = TOTAL_MUTABLE_DIM * par.mc.thread * sizeof(float);
		float* soa_ptr = (float*)calloc(1, soa_size);

		for (int i = 0; i < m.atoms.size(); i++) {
			m_ptr->atoms[i].types[0] = m.atoms[i].el;// To store 4 atoms types (el, ad, xs, sy)
			m_ptr->atoms[i].types[1] = m.atoms[i].ad;
			m_ptr->atoms[i].types[2] = m.atoms[i].xs;
			m_ptr->atoms[i].types[3] = m.atoms[i].sy;
			for (int j = 0; j < 3; j++) {
				m_ptr->atoms[i].coords[j] = m.atoms[i].coords[j];// To store atom coords
			}
		}

		// To store atoms coords and minus forces (identical for every replica)
		for (int i = 0; i < m.coords.size(); i++) {
			for (int j = 0; j < 3; j++) {
				soa_fill(soa_ptr, M_COORDS_OFFSET, i, j, 3, par.mc.thread, m.coords[i].data[j]);
				soa_fill(soa_ptr, MINUS_FORCES_OFFSET, i, j, 3, par.mc.thread, m.minus_forces[i].data[j]);
			}
		}

		// Preparing ligand data
		if(m.num_other_pairs() != 0){throw std::runtime_error("m.other_paris is not supported!");} // m.other_paris is not supported!
		if(m.ligands.size() != 1){throw std::runtime_error("Only one ligand supported!");} // Only one ligand supported!
		m_ptr->ligand.pairs.num_pairs = m.ligands[0].pairs.size();
		if (m.ligands[0].pairs.size() >= MAX_NUM_OF_LIG_PAIRS) { throw std::runtime_error("Ligand too large! The maximum number of pairs is " + 
		std::to_string(MAX_NUM_OF_LIG_PAIRS) + " and the ligand has " + std::to_string(m.ligands[0].pairs.size()) + " pairs."); }
		for (int i = 0; i < m_ptr->ligand.pairs.num_pairs; i++) {
			m_ptr->ligand.pairs.type_pair_index[i] = m.ligands[0].pairs[i].type_pair_index;
			m_ptr->ligand.pairs.a[i] = m.ligands[0].pairs[i].a;
			m_ptr->ligand.pairs.b[i] = m.ligands[0].pairs[i].b;
		}
		m_ptr->ligand.begin = m.ligands[0].begin; // 0
		m_ptr->ligand.end = m.ligands[0].end; // 29
		ligand m_ligand = m.ligands[0]; // Only support one ligand 
		if(m_ligand.end >= MAX_NUM_OF_ATOMS){throw std::runtime_error("Ligand too large! The maximum number of atoms is " +
		 std::to_string(MAX_NUM_OF_ATOMS) + " and the ligand has " + std::to_string(m_ligand.end) + " atoms.");}

		// Store root node
		m_ptr->ligand.rigid.atom_range[0][0] = m_ligand.node.begin;
		m_ptr->ligand.rigid.atom_range[0][1] = m_ligand.node.end;
		// Root node has no parent, so its relative_axis/origin are unused by
		// the kernel (which only iterates current >= 1). Zero-initialize for
		// hygiene.
		for (int i = 0; i < 3; i++) {
			m_ptr->ligand.rigid.relative_axis[0][i] = 0;
			m_ptr->ligand.rigid.relative_origin[0][i] = 0;
		}
		for (int i = 0; i < 3; i++) soa_fill(soa_ptr, ORIGIN_OFFSET, 0, i, 3, par.mc.thread, m_ligand.node.get_origin()[i]);
		for (int i = 0; i < 9; i++) soa_fill(soa_ptr, ORIENTATION_M_OFFSET, 0, i, 9, par.mc.thread, m_ligand.node.get_orientation_m().data[i]);
		soa_fill(soa_ptr, ORIENTATION_Q_OFFSET, 0, 0, 4, par.mc.thread, m_ligand.node.orientation().R_component_1());
		soa_fill(soa_ptr, ORIENTATION_Q_OFFSET, 0, 1, 4, par.mc.thread, m_ligand.node.orientation().R_component_2());
		soa_fill(soa_ptr, ORIENTATION_Q_OFFSET, 0, 2, 4, par.mc.thread, m_ligand.node.orientation().R_component_3());
		soa_fill(soa_ptr, ORIENTATION_Q_OFFSET, 0, 3, 4, par.mc.thread, m_ligand.node.orientation().R_component_4());
		for (int i = 0; i < 3; i++) soa_fill(soa_ptr, AXIS_OFFSET, 0, i, 3, par.mc.thread, 0.0f);

		struct tmp_struct {
			int start_index = 0;
			int parent_index = 0;
			void store_node(tree<segment>& child_ptr, rigid_ro_cl& rigid, float* soa_ptr, int thread_count) {
				start_index++; // start with index 1, index 0 is root node
				rigid.parent[start_index] = parent_index;
				rigid.atom_range[start_index][0] = child_ptr.node.begin;
				rigid.atom_range[start_index][1] = child_ptr.node.end;
				// relative_axis / relative_origin are read-only ligand topology;
				// store once in m_ro_cl rather than per-thread in soa_ptr.
				for (int i = 0; i < 3; i++) {
					rigid.relative_axis[start_index][i] = child_ptr.node.relative_axis[i];
					rigid.relative_origin[start_index][i] = child_ptr.node.relative_origin[i];
				}
				for (int i = 0; i < 9; i++) soa_fill(soa_ptr, ORIENTATION_M_OFFSET, start_index, i, 9, thread_count, child_ptr.node.get_orientation_m().data[i]);
				soa_fill(soa_ptr, ORIENTATION_Q_OFFSET, start_index, 0, 4, thread_count, child_ptr.node.orientation().R_component_1());
				soa_fill(soa_ptr, ORIENTATION_Q_OFFSET, start_index, 1, 4, thread_count, child_ptr.node.orientation().R_component_2());
				soa_fill(soa_ptr, ORIENTATION_Q_OFFSET, start_index, 2, 4, thread_count, child_ptr.node.orientation().R_component_3());
				soa_fill(soa_ptr, ORIENTATION_Q_OFFSET, start_index, 3, 4, thread_count, child_ptr.node.orientation().R_component_4());
				for (int i = 0; i < 3; i++) {
					soa_fill(soa_ptr, ORIGIN_OFFSET, start_index, i, 3, thread_count, child_ptr.node.get_origin()[i]);
					soa_fill(soa_ptr, AXIS_OFFSET, start_index, i, 3, thread_count, child_ptr.node.get_axis()[i]);
				}
				if (child_ptr.children.size() == 0) return;
				else {
					if (start_index >= MAX_NUM_OF_RIGID) { throw std::runtime_error("Children map too large!"); }
					int parent_index_tmp = start_index;
					for (int i = 0; i < child_ptr.children.size(); i++) {
						this->parent_index = parent_index_tmp; // Update parent index
						this->store_node(child_ptr.children[i], rigid, soa_ptr, thread_count);
					}
				}
			}
		};
		tmp_struct ts;
 		for (int i = 0; i < m_ligand.children.size(); i++) {
			ts.parent_index = 0; // Start a new branch, whose parent is 0
			ts.store_node(m_ligand.children[i], m_ptr->ligand.rigid, soa_ptr, par.mc.thread);
		}
		m_ptr->ligand.rigid.num_children = ts.start_index;

		// set children_map
		for (int i = 0; i < MAX_NUM_OF_RIGID; i++)
			for (int j = 0; j < MAX_NUM_OF_RIGID; j++)
				m_ptr->ligand.rigid.children_map[i][j] = false;
		for (int i = 1; i < m_ptr->ligand.rigid.num_children + 1; i++) {
			int parent_index = m_ptr->ligand.rigid.parent[i];
			m_ptr->ligand.rigid.children_map[parent_index][i] = true;
		}
		m_ptr->m_num_movable_atoms = m.num_movable_atoms();
		size_t m_size = sizeof(m_ro_cl);

		// Use shared pinned host buffers (allocated once before loop) instead of
		// per-ligand malloc. Ligands process sequentially so reusing the same
		// buffer is safe.
		result_coords_ptrs[ligand_count] = pinned_coords_host;
		result_ptrs[ligand_count]        = pinned_results_host;

		//cl_mem ric_gpu;
		CreateDeviceBuffer(&ric_gpus[ligand_count], CL_MEM_READ_ONLY, ric_size, context);
		err = clEnqueueWriteBuffer(queue, ric_gpus[ligand_count], false, 0, ric_size, ric_ptr, 0, NULL, NULL); checkErr(err);

		//cl_mem m_gpu;
		CreateDeviceBuffer(&m_gpus[ligand_count], CL_MEM_READ_ONLY, m_size, context);
		err = clEnqueueWriteBuffer(queue, m_gpus[ligand_count], false, 0, m_size, m_ptr, 0, NULL, NULL); checkErr(err);

		CreateDeviceBuffer(&m_soa_gpus[ligand_count], CL_MEM_READ_WRITE, soa_size, context);
		err = clEnqueueWriteBuffer(queue, m_soa_gpus[ligand_count], false, 0, soa_size, soa_ptr, 0, NULL, NULL); checkErr(err);


		//cl_mem random_maps_gpu;
		CreateDeviceBuffer(&random_maps_gpus[ligand_count], CL_MEM_READ_ONLY, rand_maps_size, context);
		err = clEnqueueWriteBuffer(queue, random_maps_gpus[ligand_count], false, 0, rand_maps_size, rand_maps_ptr, 0, NULL, NULL); checkErr(err);

		//cl_mem result_coords_gpu;
		CreateDeviceBuffer(&result_coords_gpus[ligand_count], CL_MEM_WRITE_ONLY, mis_ptr->thread * sizeof(ligand_atom_coords_cl), context);
		err = clEnqueueWriteBuffer(queue, result_coords_gpus[ligand_count], false, 0, mis_ptr->thread * sizeof(ligand_atom_coords_cl),
			result_coords_ptrs[ligand_count], 0, NULL, NULL); checkErr(err);

		//cl_mem results_gpu;
		CreateDeviceBuffer(&result_gpus[ligand_count], CL_MEM_WRITE_ONLY, par.mc.thread * sizeof(output_type_cl), context);
		err = clEnqueueWriteBuffer(queue, result_gpus[ligand_count], false, 0, par.mc.thread * sizeof(output_type_cl),
			result_ptrs[ligand_count], 0, NULL, NULL); checkErr(err);

		clFinish(queue);

		SetKernelArg(kernels[1], 0, sizeof(cl_mem), &ric_gpus[ligand_count]);
		SetKernelArg(kernels[1], 1, sizeof(cl_mem), &m_gpus[ligand_count]);
		SetKernelArg(kernels[1], 2, sizeof(cl_mem), &pre_gpu);
		SetKernelArg(kernels[1], 3, sizeof(cl_mem), &grids_gpu);
		SetKernelArg(kernels[1], 4, sizeof(cl_mem), &random_maps_gpus[ligand_count]);
		SetKernelArg(kernels[1], 5, sizeof(cl_mem), &result_coords_gpus[ligand_count]);
		SetKernelArg(kernels[1], 6, sizeof(cl_mem), &result_gpus[ligand_count]);
		SetKernelArg(kernels[1], 7, sizeof(cl_mem), &mis_gpu);
		SetKernelArg(kernels[1], 8, sizeof(int),	&torsion_sizes[ligand_count]);
		SetKernelArg(kernels[1], 9, sizeof(int),	&par.mc.search_depth[ligand_count]);
		SetKernelArg(kernels[1], 10, sizeof(int),	&par.mc.ssd_par.bfgs_steps[ligand_count]);
		SetKernelArg(kernels[1], 11, sizeof(int),	&rilc_bfgs);
		SetKernelArg(kernels[1], 12, sizeof(cl_mem), &m_soa_gpus[ligand_count]);

		// 1D launch sized to mis_ptr->total_wi (set above to ceil(thread/32)*32).
		// kernel2 only uses get_global_linear_id(), so dimensionality doesn't
		// matter; gx/gy/gs at the top of kernel2 are dead.
		size_t kernel2_global_size = mis_ptr->total_wi;
		size_t kernel2_local_size = 32;

		err = clEnqueueNDRangeKernel(queue, kernels[1], 1, 0, &kernel2_global_size, &kernel2_local_size,
			0, NULL, &ligands_events[ligand_count]); checkErr(err);

		clWaitForEvents(1, &ligands_events[ligand_count]);

		err = clEnqueueReadBuffer(queue, result_gpus[ligand_count], false, 0, par.mc.thread * sizeof(output_type_cl),
			result_ptrs[ligand_count], 0, NULL, NULL); checkErr(err);

		err = clEnqueueReadBuffer(queue, result_coords_gpus[ligand_count], false, 0, par.mc.thread * sizeof(ligand_atom_coords_cl),
			result_coords_ptrs[ligand_count], 0, NULL, NULL); checkErr(err);

		clFinish(queue);

		std::vector<output_type> result_vina = cl_to_vina(result_ptrs[ligand_count], result_coords_ptrs[ligand_count],
			par.mc.thread, torsion_sizes[ligand_count]);
		// if empty, something goes wrong in the device part
		if (result_vina.size() == 0) {
			status = ABORT;
# ifdef NDEBUG
			console_thread.join();
# endif
			exit(-1);
		}

		// Write back to vina
		for (int i = 0; i < par.mc.thread; i++) {
			//assert(result_vina[i].coords.size()== mis_ptr->)
			add_to_output_container(outs[ligand_count], result_vina[i], par.mc.min_rmsd, par.mc.num_saved_mins);
		}
		// VINA_CHECK(!outs[ligand_count].empty());
		// VINA_CHECK(outs[ligand_count].front().e <= outs[ligand_count].back().e);

		free(rand_maps_ptrs[ligand_count]);
		free(ric_ptrs[ligand_count]);
		free(m_ptrs[ligand_count]);
		free(soa_ptr);
		// result_coords_ptrs and result_ptrs point into shared pinned buffers
		// (cleaned up after the ligand loop); do not free them here.


		err = clReleaseMemObject(result_coords_gpus[ligand_count]); checkErr(err);
		err = clReleaseMemObject(result_gpus[ligand_count]);		checkErr(err);
		err = clReleaseMemObject(ric_gpus[ligand_count]);			checkErr(err);
		err = clReleaseMemObject(m_gpus[ligand_count]);				checkErr(err);
		err = clReleaseMemObject(m_soa_gpus[ligand_count]);			checkErr(err);
		err = clReleaseMemObject(random_maps_gpus[ligand_count]);	checkErr(err);
		
#ifndef TIME_ANALYSIS
		err = clReleaseEvent(ligands_events[ligand_count]);			checkErr(err);
#endif // !TIME_ANALYSIS
	}
	catch(const std::exception& e){
		std::cerr << "Error processing ligand " << ligand_count << ": " << ligand_names[ligand_count][0] << std::endl;
		std::cerr << "Exception: " << e.what() << std::endl;
		std::cerr << "Skipping this ligand and continuing with the next one." << std::endl;
		continue;
	}
	}

	// Tear down pinned host buffers (Tier 1d).
	err = clEnqueueUnmapMemObject(queue, pinned_results_buf, pinned_results_host, 0, NULL, NULL); checkErr(err);
	err = clEnqueueUnmapMemObject(queue, pinned_coords_buf,  pinned_coords_host,  0, NULL, NULL); checkErr(err);
	clFinish(queue);
	err = clReleaseMemObject(pinned_results_buf); checkErr(err);
	err = clReleaseMemObject(pinned_coords_buf);  checkErr(err);

	free(mis_ptr);
	err = clReleaseMemObject(mis_gpu);		checkErr(err);
	err = clReleaseMemObject(grids_gpu);	checkErr(err);
	err = clReleaseMemObject(pre_gpu);		checkErr(err);

	status = FINISH;
# ifdef NDEBUG
	console_thread.join(); // wait the thread finish
# endif
#ifdef TIME_ANALYSIS
	// Output Analysis
	cl_ulong time_start, time_end;
	double total_time = 0;
	std::string log_file = vina_gpu_log_path();
	// delete file if file exists
	if (std::ifstream(log_file)) {
		std::remove(log_file.c_str());
	}
	for (int ligand_count = 0; ligand_count < ms.size(); ligand_count++) {
		cl_event event_tmp = ligands_events[ligand_count];
		err = clGetEventProfilingInfo(event_tmp, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); checkErr(err);
		err = clGetEventProfilingInfo(event_tmp, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); checkErr(err);
		total_time = time_end - time_start;
		err = clReleaseEvent(ligands_events[ligand_count]);			checkErr(err);
		printf("\nAutoDockVina-GPU3 ligand %d runtime = %0.3f s", ligand_count,(total_time / 1000000000.0));

		std::cout << std::setiosflags(std::ios::fixed);
		std::ofstream file(log_file, std::ios::app);
		if (file.is_open())
		{
			file << "AutoDockVina-GPU3 monte carlo runtime of ligand " << ligand_names[ligand_count][0] << " = " << std::setprecision(5) << (total_time / 1000000000.0) << " s" << std::endl;
			file.close();
		}
	}
	printf("\n");
#endif
#ifdef OPENCL_TIME_ANALYSIS
		cout << setiosflags(ios::fixed);
		std::ofstream file(vina_gpu_log_path());
		if (file.is_open())
		{
			file << "GPU grid cache runtime = " << (kernel1_total_time / 1000000000.0) << " s" << std::endl;
			file << "GPU monte carlo runtime = " << (kernel2_total_time / 1000000000.0) << " s" << std::endl;
			file.close();
		}
#endif
}